"""Baseline trainer with test-matched image augmentation."""
import argparse
import copy
import glob
import io
import itertools
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'FIT5215_Dataset'
TEST_DIR = ROOT / 'data' / 'test_set' / 'test_dataset'
TEMPLATE = ROOT / 'data' / 'test_set' / 'my_solution.csv'
SEED, NATIVE, SIZE = 1234, 64, 128
CAND_SCALES = (96, 112, 128, 144, 160)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLASS_FIX = {'butterfiles': 'butterflies'}


def to_test_domain(img, quality=75, size=NATIVE):
    """Resize and JPEG-encode an image like test data."""
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


NORM = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]


class TrainTF:
    def __init__(self):
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.03),
            *NORM,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])

    def __call__(self, img, native):
        if random.random() < 0.12:
            img = ImageOps.grayscale(img).convert('RGB')
        if not native:
            size = NATIVE if random.random() < 0.7 else random.randint(56, 72)
            img = to_test_domain(img, random.randint(65, 85), size)
        return self.aug(img)


class EvalTF:
    def __init__(self, scale):
        self.tf = transforms.Compose([transforms.Resize((scale, scale)), *NORM])

    def __call__(self, img, native):
        return self.tf(img if native else to_test_domain(img))


class ItemSet(Dataset):
    """Dataset entries: path, label, native flag."""

    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y, nat = self.items[i]
        return self.tf(Image.open(p).convert('RGB'), nat), y


def mix_batch(x, y, n_cls):
    lam = np.random.beta(1.0, 1.0)
    perm = torch.randperm(x.size(0), device=x.device)
    if np.random.rand() < 0.5:
        x = lam * x + (1 - lam) * x[perm]
    else:
        H, W = x.shape[2:]
        r = math.sqrt(1 - lam)
        cy, cx = np.random.randint(H), np.random.randint(W)
        ch, cw = int(H * r), int(W * r)
        y1, y2 = np.clip(cy - ch // 2, 0, H), np.clip(cy + ch // 2, 0, H)
        x1, x2 = np.clip(cx - cw // 2, 0, W), np.clip(cx + cw // 2, 0, W)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    t = torch.zeros(x.size(0), n_cls, device=x.device)
    t.scatter_(1, y.view(-1, 1), lam)
    t.scatter_add_(1, y[perm].view(-1, 1), torch.full_like(t[:, :1], 1 - lam))
    return x, t


def soft_ce(logits, t, smooth=0.1):
    t = t * (1 - smooth) + smooth / logits.size(1)
    return -(t * torch.log_softmax(logits, 1)).sum(1).mean()


class EMA:
    def __init__(self, model, decay=0.998):
        self.m, self.decay = copy.deepcopy(model).eval(), decay
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(p, alpha=1 - self.decay)
            else:
                e.copy_(p)


def build(arch, n_cls, pretrained):
    m = models.get_model(arch, weights='DEFAULT' if pretrained else None)
    if hasattr(m, 'fc'):
        m.fc = nn.Linear(m.fc.in_features, n_cls)
    else:
        i = len(m.classifier) - 1
        m.classifier[i] = nn.Linear(m.classifier[i].in_features, n_cls)
    return m


@torch.no_grad()
def probs_at(model, items, dev, scale, batch=128):
    """Return flip-averaged probabilities at one scale."""
    model.eval()
    out = []
    for x, _ in DataLoader(ItemSet(items, EvalTF(scale)), batch_size=batch, num_workers=4):
        x = x.to(dev).float()
        out.append(((torch.softmax(model(x), 1) +
                     torch.softmax(model(torch.flip(x, [3])), 1)) / 2).cpu())
    return torch.cat(out).numpy()


def sinkhorn(P, target, tau=1.0, iters=300):
    P = P / P.sum(1, keepdims=True)
    logv = np.zeros(P.shape[1])
    for _ in range(iters):
        Q = P * np.exp(logv)
        Q /= Q.sum(1, keepdims=True)
        logv += np.log(target) - np.log(Q.sum(0) + 1e-12)
    Q = P * np.exp(tau * logv)
    return Q / Q.sum(1, keepdims=True)


def write_sub(P, paths, classes, name):
    ids = [int(Path(p).name.split('_', 1)[0]) for p in paths]
    lab = [CLASS_FIX.get(classes[i], classes[i]) for i in P.argmax(1)]
    sub = pd.DataFrame({'ID': ids, 'Label': lab}).sort_values('ID').reset_index(drop=True)
    assert sub['ID'].tolist() == pd.read_csv(TEMPLATE)['ID'].tolist()
    sub.to_csv(ROOT / name, index=False)
    vc = sub['Label'].value_counts()
    print(f'  wrote {name}: max {vc.max()} ({vc.idxmax()})  min {vc.min()} ({vc.idxmin()})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='resnet34',
                    choices=['resnet34', 'resnet50', 'convnext_tiny', 'efficientnet_v2_s'])
    ap.add_argument('--epochs', type=int, default=35)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--pseudo', default=None, help='probs npz from a previous run')
    ap.add_argument('--pseudo-thresh', type=float, default=0.85)
    ap.add_argument('--predict-only', action='store_true')
    args = ap.parse_args()
    tag = f'v4_{args.arch}' + (f'_s{args.seed}' if args.seed != SEED else '') + ('_pl' if args.pseudo else '')
    ckpt = ROOT / f'{tag}.pt'

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available()
                       else 'mps' if torch.backends.mps.is_available() else 'cpu')

    base = datasets.ImageFolder(DATA_DIR)
    classes, n_cls = base.classes, len(base.classes)
    items = [(p, y, Image.open(p).size == (NATIVE, NATIVE)) for p, y in base.samples]
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(len(items), generator=g).tolist()
    cut = int(0.9 * len(items))
    tr, va = [items[i] for i in idx[:cut]], [items[i] for i in idx[cut:]]
    yv = np.array([y for _, y, _ in va])
    paths = sorted(glob.glob(str(TEST_DIR / '*_animal.jpeg')))
    assert len(paths) == 11681
    test_items = [(p, 0, True) for p in paths]
    counts = np.bincount([y for _, y, _ in items], minlength=n_cls).astype(float)
    prior = counts / counts.sum() * len(paths)
    print(f'{tag} | device {dev} | {len(tr)} train ({sum(n for *_, n in tr)} native) / {len(va)} val')

    if args.pseudo:
        P = np.load(args.pseudo)['test']
        P = sinkhorn(P, prior, 0.7)
        keep = P.max(1) >= args.pseudo_thresh
        tr = tr + [(paths[i], int(P[i].argmax()), True) for i in np.where(keep)[0]]
        print(f'pseudo-labels: kept {keep.sum()} / {len(paths)}  ->  {len(tr)} train items')

    model = build(args.arch, n_cls, pretrained=not args.predict_only).to(dev)

    if args.predict_only:
        model.load_state_dict(torch.load(ckpt, map_location=dev))
    else:
        loader = DataLoader(ItemSet(tr, TrainTF()), batch_size=args.batch, shuffle=True,
                            drop_last=True, num_workers=6, persistent_workers=True)
        val128 = [(p, y, n) for p, y, n in va]
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        steps = args.epochs * len(loader)
        warm = len(loader)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
        ema, best = EMA(model), -1.0
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            for x, y in loader:
                x, y = x.to(dev).float(), y.to(dev)
                if np.random.rand() < 0.5:
                    x, t = mix_batch(x, y, n_cls)
                else:
                    t = torch.zeros(x.size(0), n_cls, device=dev).scatter_(1, y.view(-1, 1), 1.0)
                opt.zero_grad(set_to_none=True)
                soft_ce(model(x), t).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); sched.step(); ema.update(model)
            a_raw = (probs_at(model, val128, dev, SIZE).argmax(1) == yv).mean()
            a_ema = (probs_at(ema.m, val128, dev, SIZE).argmax(1) == yv).mean()
            mark = ''
            if max(a_raw, a_ema) > best:
                best, mark = max(a_raw, a_ema), '  *'
                torch.save((ema.m if a_ema >= a_raw else model).state_dict(), ckpt)
            print(f'ep {ep:>2}/{args.epochs} | val {a_raw*100:5.2f}% | ema {a_ema*100:5.2f}% '
                  f'| best {best*100:5.2f}% | {time.time()-t0:.0f}s{mark}')
        model.load_state_dict(torch.load(ckpt, map_location=dev))

    print('\nTTA selection on validation:')
    pv = {s: probs_at(model, va, dev, s) for s in CAND_SCALES}
    best_combo, best_acc = None, -1
    for k in (1, 2, 3):
        for combo in itertools.combinations(CAND_SCALES, k):
            a = (sum(pv[s] for s in combo).argmax(1) == yv).mean()
            if a > best_acc:
                best_combo, best_acc = combo, a
    for s in CAND_SCALES:
        print(f'  {s:>3}px alone  {(pv[s].argmax(1) == yv).mean()*100:5.2f}%')
    print(f'  best combo {best_combo}  {best_acc*100:.2f}%')
    Pv = sum(pv[s] for s in best_combo) / len(best_combo)

    vt = np.bincount(yv, minlength=n_cls).astype(float)
    taus = {t: (sinkhorn(Pv, vt, t).argmax(1) == yv).mean() for t in (0.0, 0.3, 0.5, 0.7, 1.0)}
    for t, a in taus.items():
        print(f'  sinkhorn tau {t:.1f}  {a*100:5.2f}%')
    tau = max(taus, key=taus.get)

    Pt = sum(probs_at(model, test_items, dev, s) for s in best_combo) / len(best_combo)
    np.savez_compressed(ROOT / f'probs_{tag}.npz', test=Pt, val=Pv, yv=yv, scales=best_combo)
    print(f'\nsaved probs_{tag}.npz  (val {best_acc*100:.2f}%, tau {tau})')
    write_sub(Pt, paths, classes, f'sub_{tag}_raw.csv')
    if tau > 0:
        write_sub(sinkhorn(Pt, prior, tau), paths, classes, f'sub_{tag}_tau{int(tau*10)}.csv')


if __name__ == '__main__':
    main()
