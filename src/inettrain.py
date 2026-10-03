"""Fine-tune ImageNet models by grouping 1,000 logits into 20 class scores."""
import argparse, copy, math, random, time
import numpy as np, timm, torch, torch.nn as nn
from torchvision import transforms
from timm.optim import create_optimizer_v2
from torch.utils.data import DataLoader
import common as C
import kaggle_clip as kc
from kaggle_v4 import soft_ce, write_sub
from retrain import SoftSet, mix_soft
from dinotrain import TrainTF as _TrainTF, EvalTF as _EvalTF
from inet import GROUP

class Net(nn.Module):
    def __init__(self, backbone, groups):
        super().__init__(); self.backbone = backbone
        self.register_buffer("mask", torch.full((len(groups), 1000), -1e4))
        for c, idx in enumerate(groups): self.mask[c, idx] = 0.0
    def forward(self, x):
        z = self.backbone(x).float()
        return torch.logsumexp(z[:, None, :] + self.mask[None], dim=2)

def _renorm(compose, mean, std):
    """Set the transform normalization."""
    for i, t in enumerate(compose.transforms):
        if isinstance(t, transforms.Normalize):
            compose.transforms[i] = transforms.Normalize(mean, std)
    return compose


def make_tf(res, mean, std):
    class TrainTF(_TrainTF):
        def __init__(self):
            super().__init__(res); _renorm(self.aug, mean, std)

    class EvalTF(_EvalTF):
        def __init__(self, crop=1.0):
            super().__init__(res, crop); _renorm(self.norm, mean, std)

    return TrainTF, EvalTF


@torch.no_grad()
def probs_tta(model, items, dev, EvalTF, batch=48):
    model.eval(); outs = []
    for crop, w in ((1.0, 2), (0.92, 2), (0.85, 1)):
        acc = []
        for x, _ in DataLoader(kc.ItemSet(items, EvalTF(crop)), batch_size=batch, num_workers=8):
            x = x.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.softmax(model(x), 1)
                if w == 2: p = (p + torch.softmax(model(torch.flip(x, [3])), 1)) / 2
            acc.append(p.cpu())
        outs.append((torch.cat(acc), w))
    return (sum(o * w for o, w in outs) / sum(w for _, w in outs)).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--model", required=True); ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--pseudo", default="probs_w6evab_a06.npz"); ap.add_argument("--pseudo-frac", type=float, default=0.90)
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6); ap.add_argument("--layer-decay", type=float, default=0.85)
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--seed", type=int, default=91); ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed); dev = torch.device("cuda")
    items, classes = C.train_items(); K = len(classes); groups = [GROUP[c] for c in classes]
    order = C.split_order(len(items)); vs = set(order.tolist())
    tr = [it for i, it in enumerate(items) if i not in vs]; va = [items[i] for i in order]
    yv = np.array([y for _, y, _ in va]); paths = C.test_paths(); test_items = [(p, 0, True) for p in paths]
    onehot = lambda c: np.eye(K, dtype=np.float32)[c]
    P = np.load(C.ROOT / args.pseudo)["test"]; lab, conf = P.argmax(1), P.max(1); chosen = []
    for c in range(K):
        m = np.where(lab == c)[0]; m = m[np.argsort(-conf[m])]; chosen += m[: int(len(m) * args.pseudo_frac)].tolist()
    rows = [(p, onehot(y), nat) for p, y, nat in tr] + [(paths[i], onehot(int(lab[i])), True) for i in chosen]
    backbone = timm.create_model(args.model, pretrained=True)
    cfg = timm.data.resolve_data_config(model=backbone); TrainTF, EvalTF = make_tf(args.res, cfg["mean"], cfg["std"])
    model = Net(backbone, groups).to(dev); anchor = copy.deepcopy(model.state_dict())
    print(f"{args.tag} | {args.model} | {len(tr)} train + {len(chosen)} pseudo / {len(va)} val | res {args.res}", flush=True)
    Pv = probs_tta(model, va, dev, EvalTF); print(f"  anchor (ImageNet weights)  val {(Pv.argmax(1) == yv).mean() * 100:5.2f}%", flush=True)
    loader = DataLoader(SoftSet(rows, TrainTF()), batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers, persistent_workers=True)
    opt = create_optimizer_v2(model.backbone, opt="adamw", lr=args.lr, weight_decay=0.05, layer_decay=args.layer_decay)
    steps, warm = args.epochs * len(loader), len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    ema = kc.EMA(model); ckpt = C.ROOT / f"clip_{args.tag}_last.pt"
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); model.train()
        for x, t in loader:
            x, t = x.to(dev), t.to(dev)
            if np.random.rand() < 0.3: x, t = mix_soft(x, t)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): loss = soft_ce(model(x), t)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); ema.update(model)
        Pv = probs_tta(ema.m, va, dev, EvalTF); torch.save(ema.m.state_dict(), ckpt)
        print(f"ep {ep:>2}/{args.epochs} | ema val {(Pv.argmax(1) == yv).mean() * 100:5.2f}% | loss {loss.item():.3f} | {time.time() - t0:.0f}s", flush=True)
    ft = torch.load(ckpt, map_location=dev); print("\nWiSE ladder ImageNet -> student:", flush=True)
    for a in args.alphas:
        model.load_state_dict(kc.interpolate(anchor, ft, a))
        Pv, Pt = probs_tta(model, va, dev, EvalTF), probs_tta(model, test_items, dev, EvalTF)
        name = f"{args.tag}_a{int(round(a * 10)):02d}"
        np.savez_compressed(C.ROOT / f"probs_{name}.npz", test=Pt, val=Pv, yv=yv, scales=(args.res,)); write_sub(Pt, paths, classes, f"sub_{name}.csv")
        print(f"  {name:20s} val {(Pv.argmax(1) == yv).mean() * 100:5.2f}%  conf<0.5 {(Pt.max(1) < 0.5).mean() * 100:4.1f}%", flush=True)
    ckpt.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
