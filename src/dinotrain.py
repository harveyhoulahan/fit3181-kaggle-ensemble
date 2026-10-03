"""Fine-tune a DINO model and emit WiSE-FT probability files."""
import argparse, copy, math, random, time
import numpy as np, timm, torch, torch.nn as nn
from timm.optim import create_optimizer_v2
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import common as C
import kaggle_clip as kc
from kaggle_v4 import soft_ce, to_test_domain, write_sub
from retrain import SoftSet, mix_soft

IN_MEAN, IN_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class TrainTF:
    def __init__(self, res):
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(res, scale=(0.7, 1.0), ratio=(0.85, 1.18), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(), transforms.TrivialAugmentWide(),
            transforms.ToTensor(), transforms.Normalize(IN_MEAN, IN_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12))])
    def __call__(self, img, native):
        if random.random() < 0.12:
            from PIL import ImageOps; img = ImageOps.grayscale(img).convert("RGB")
        if not native:
            size = C.NATIVE if random.random() < 0.7 else random.randint(56, 72)
            img = to_test_domain(img, random.randint(65, 85), size)
        return self.aug(img)

class EvalTF:
    def __init__(self, res, crop=1.0):
        self.res, self.crop = res, crop
        self.norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IN_MEAN, IN_STD)])
    def __call__(self, img, native):
        im = img if native else to_test_domain(img)
        if self.crop < 1.0:
            w, h = im.size; dw, dh = int(w * (1 - self.crop) / 2), int(h * (1 - self.crop) / 2)
            im = im.crop((dw, dh, w - dw, h - dh))
        return self.norm(im.resize((self.res, self.res), Image.BICUBIC))

class Net(nn.Module):
    def __init__(self, backbone, W, b, scale=30.0):
        super().__init__()
        self.backbone, self.scale = backbone, scale
        self.W, self.b = nn.Parameter(W.clone()), nn.Parameter(b.clone())
    def forward(self, x):
        f = self.backbone(x); f = f / f.norm(dim=-1, keepdim=True)
        return self.scale * f @ self.W + self.b

@torch.no_grad()
def probs_tta(model, items, dev, res, batch=96):
    model.eval(); outs = []
    for crop, w in ((1.0, 2), (0.92, 2), (0.85, 1)):
        acc = []
        for x, _ in DataLoader(kc.ItemSet(items, EvalTF(res, crop)), batch_size=batch, num_workers=8):
            x = x.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.softmax(model(x).float(), 1)
                if w == 2: p = (p + torch.softmax(model(torch.flip(x, [3])).float(), 1)) / 2
            acc.append(p.cpu())
        outs.append((torch.cat(acc), w))
    return (sum(o * w for o, w in outs) / sum(w for _, w in outs)).numpy()

def fit_probe(X, T, dev, scale=30.0, epochs=300, lr=0.1, wd=1e-4):
    X = torch.tensor(X, device=dev); T = torch.tensor(T, device=dev)
    W = torch.zeros(X.shape[1], T.shape[1], device=dev, requires_grad=True); b = torch.zeros(T.shape[1], device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad(); (-(T * torch.log_softmax(X @ W * scale + b, 1)).sum(1).mean()).backward(); opt.step()
    return W.detach(), b.detach()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--model", default="vit_large_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--feat", default="dinoL"); ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--pseudo", default="probs_w6evab_a06.npz"); ap.add_argument("--pseudo-frac", type=float, default=0.90)
    ap.add_argument("--epochs", type=int, default=7); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--layer-decay", type=float, default=0.85)
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
    ap.add_argument("--seed", type=int, default=71); ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev = torch.device("cuda")
    items, classes = C.train_items(); K = len(classes)
    order = C.split_order(len(items)); vs = set(order.tolist())
    tr = [it for i, it in enumerate(items) if i not in vs]; va = [items[i] for i in order]
    yv = np.array([y for _, y, _ in va]); paths = C.test_paths(); test_items = [(p, 0, True) for p in paths]
    onehot = lambda c: np.eye(K, dtype=np.float32)[c]

    P = np.load(C.ROOT / args.pseudo)["test"]; lab, conf = P.argmax(1), P.max(1); chosen = []
    for c in range(K):
        m = np.where(lab == c)[0]; m = m[np.argsort(-conf[m])]; chosen += m[: int(len(m) * args.pseudo_frac)].tolist()
    rows = [(p, onehot(y), nat) for p, y, nat in tr] + [(paths[i], onehot(int(lab[i])), True) for i in chosen]
    print(f"{args.tag} | {args.model} | {len(tr)} train + {len(chosen)} pseudo ({args.pseudo}) / {len(va)} val", flush=True)

    # Fit the frozen-backbone interpolation anchor.
    F = np.load(C.CACHE / f"feat_{args.feat}.npz"); Xtr = F["train"].astype(np.float32); Xte = F["test"].astype(np.float32)
    nrm = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True); Xtr, Xte = nrm(Xtr), nrm(Xte)
    vm = np.zeros(len(items), bool); vm[order] = True; ytr = F["y"]
    W0, b0 = fit_probe(np.concatenate([Xtr[~vm], Xte[chosen]]), np.concatenate([np.eye(K, dtype=np.float32)[ytr[~vm]], np.eye(K, dtype=np.float32)[lab[chosen]]]), dev)
    backbone = timm.create_model(args.model, pretrained=True, num_classes=0, img_size=args.res)
    model = Net(backbone, W0, b0).to(dev)
    anchor = copy.deepcopy(model.state_dict())
    Pv = probs_tta(model, va, dev, args.res); print(f"  anchor (frozen+probe)  val {(Pv.argmax(1) == yv).mean() * 100:5.2f}%", flush=True)

    loader = DataLoader(SoftSet(rows, TrainTF(args.res)), batch_size=args.batch, shuffle=True, drop_last=True,
                        num_workers=args.workers, persistent_workers=True)
    opt = create_optimizer_v2(model.backbone, opt="adamw", lr=args.lr, weight_decay=0.05, layer_decay=args.layer_decay)
    opt.add_param_group({"params": [model.W, model.b], "lr": args.lr * 20, "weight_decay": 0.0})
    steps, warm = args.epochs * len(loader), len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    ema = kc.EMA(model); ckpt = C.ROOT / f"clip_{args.tag}_last.pt"
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); model.train()
        for x, t in loader:
            x, t = x.to(dev), t.to(dev)
            if np.random.rand() < 0.3: x, t = mix_soft(x, t)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): loss = soft_ce(model(x).float(), t)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); ema.update(model)
        Pv = probs_tta(ema.m, va, dev, args.res)
        torch.save(ema.m.state_dict(), ckpt)
        print(f"ep {ep:>2}/{args.epochs} | ema val {(Pv.argmax(1) == yv).mean() * 100:5.2f}% | loss {loss.item():.3f} | {time.time() - t0:.0f}s", flush=True)

    ft = torch.load(ckpt, map_location=dev)
    print("\nWiSE ladder anchor -> student:", flush=True)
    for a in args.alphas:
        model.load_state_dict(kc.interpolate(anchor, ft, a))
        Pv, Pt = probs_tta(model, va, dev, args.res), probs_tta(model, test_items, dev, args.res)
        name = f"{args.tag}_a{int(round(a * 10)):02d}"
        np.savez_compressed(C.ROOT / f"probs_{name}.npz", test=Pt, val=Pv, yv=yv, scales=(args.res,))
        write_sub(Pt, paths, classes, f"sub_{name}.csv")
        print(f"  {name:20s} val {(Pv.argmax(1) == yv).mean() * 100:5.2f}%  conf<0.5 {(Pt.max(1) < 0.5).mean() * 100:4.1f}%", flush=True)

if __name__ == "__main__":
    main()
