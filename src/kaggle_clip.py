"""Train a CLIP member and emit WiSE-FT probability files."""
import argparse
import copy
import glob
import io
import math
import random
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from kaggle_v4 import (DATA_DIR, TEST_DIR, NATIVE, SEED, mix_batch, sinkhorn, soft_ce,
                       to_test_domain, write_sub)

ROOT = Path(__file__).resolve().parent.parent
RES = 160
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
NAMES = {'butterfiles': 'butterfly', 'fishes': 'fish', 'handguns': 'handgun', 'lipsticks': 'lipstick',
         'breads': 'bread', 'cakes': 'cake', 'bottles': 'bottle', 'vases': 'vase', 'birds': 'bird',
         'cats': 'cat', 'chickens': 'chicken', 'cows': 'cow', 'dogs': 'dog', 'ducks': 'duck',
         'elephants': 'elephant', 'horses': 'horse', 'lions': 'lion', 'seals': 'seal',
         'snakes': 'snake', 'spiders': 'spider'}
TEMPLATES = ['a photo of a {}.', 'a drawing of a {}.', 'a painting of a {}.', 'a sketch of a {}.',
             'a 3d render of a {}.', 'a cartoon {}.', 'a low resolution photo of a {}.',
             'a black and white photo of a {}.', 'a close-up photo of a {}.', 'art of a {}.']

NORM = [transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)]


class TrainTF:
    def __init__(self):
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(RES, scale=(0.7, 1.0), ratio=(0.85, 1.18),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            *NORM,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
        ])

    def __call__(self, img, native):
        if random.random() < 0.12:
            img = ImageOps.grayscale(img).convert('RGB')
        if not native:
            size = NATIVE if random.random() < 0.7 else random.randint(56, 72)
            img = to_test_domain(img, random.randint(65, 85), size)
        return self.aug(img)


class EvalTF:
    def __init__(self):
        self.tf = transforms.Compose([transforms.Resize((RES, RES), interpolation=transforms.InterpolationMode.BICUBIC), *NORM])

    def __call__(self, img, native):
        return self.tf(img if native else to_test_domain(img))


class ItemSet(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y, nat = self.items[i]
        return self.tf(Image.open(p).convert('RGB'), nat), y


class Clf(nn.Module):
    """CLIP visual tower with a cosine text head."""

    def __init__(self, visual, text_w, scale=100.0):
        super().__init__()
        self.visual = visual
        self.head = nn.Parameter(text_w.clone())
        self.scale = scale

    def forward(self, x):
        f = self.visual(x)
        f = f / f.norm(dim=-1, keepdim=True)
        w = self.head / self.head.norm(dim=-1, keepdim=True)
        return self.scale * f @ w.T


class EMA:
    def __init__(self, model, decay=0.998):
        self.m, self.decay = copy.deepcopy(model).eval(), decay
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            e.mul_(self.decay).add_(p, alpha=1 - self.decay) if e.dtype.is_floating_point else e.copy_(p)


@torch.no_grad()
def probs(model, items, dev, batch=128):
    model.eval()
    out = []
    for x, _ in DataLoader(ItemSet(items, EvalTF()), batch_size=batch, num_workers=4):
        x = x.to(dev).float()
        out.append(((torch.softmax(model(x), 1) + torch.softmax(model(torch.flip(x, [3])), 1)) / 2).cpu())
    return torch.cat(out).numpy()


def balance(P, prior):
    """Return prediction-prior histogram distance."""
    h = np.bincount(P.argmax(1), minlength=len(prior)) / len(P)
    return 0.5 * np.abs(h - prior / prior.sum()).sum()


def interpolate(zs, ft, alpha):
    """Linearly interpolate model states."""
    return {k: (1 - alpha) * zs[k] + alpha * ft[k] if ft[k].dtype.is_floating_point else ft[k] for k in ft}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ViT-B-16', choices=['ViT-B-16', 'ViT-L-14'])
    ap.add_argument('--pretrained', default='openai')
    ap.add_argument('--res', type=int, default=160)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--pseudo', default=None, help='probability file for pseudo-labels')
    ap.add_argument('--pseudo-frac', type=float, default=0.6,
                    help='confident fraction kept per class')
    args = ap.parse_args()
    global RES
    RES = args.res
    # OpenAI weights require QuickGELU.
    arch = args.model + ('-quickgelu' if args.pretrained == 'openai' else '')
    tag = f'clip_{args.model.replace("-", "").lower()}' + ('_pl' if args.pseudo else '')

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
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
    vt = np.bincount(yv, minlength=n_cls).astype(float)
    paths = sorted(glob.glob(str(TEST_DIR / '*_animal.jpeg')))
    test_items = [(p, 0, True) for p in paths]
    counts = np.bincount([y for _, y, _ in items], minlength=n_cls).astype(float)
    prior = counts / counts.sum() * len(paths)

    clip, _, _ = open_clip.create_model_and_transforms(arch, pretrained=args.pretrained, force_image_size=RES)
    tok = open_clip.get_tokenizer(arch)
    with torch.no_grad():
        W = []
        for c in classes:
            e = clip.encode_text(tok([t.format(NAMES[c]) for t in TEMPLATES]))
            e = e / e.norm(dim=-1, keepdim=True)
            W.append(e.mean(0))
        W = torch.stack(W)
        W = W / W.norm(dim=-1, keepdim=True)
    model = Clf(clip.visual, W).to(dev)
    del clip
    zs_state = copy.deepcopy(model.state_dict())

    def report(name, m):
        Pv, Pt = probs(m, va, dev), probs(m, test_items, dev)
        print(f'  {name:14s} val {(Pv.argmax(1) == yv).mean()*100:5.2f}%  test balance-error {balance(Pt, prior):.3f}  '
              f'test conf<0.5 {(Pt.max(1) < 0.5).mean()*100:4.1f}%')
        return Pv, Pt

    print(f'{tag} | device {dev} | {len(tr)} train / {len(va)} val')
    Pv0, Pt0 = report('zero-shot', model)
    np.savez_compressed(ROOT / f'probs_{tag}_zs.npz', test=Pt0, val=Pv0, yv=yv, scales=(RES,))
    write_sub(Pt0, paths, classes, f'sub_{tag}_zeroshot.csv')

    if args.pseudo:
        # Keep the most confident predictions in each class.
        P = np.load(args.pseudo)['test']
        lab, conf = P.argmax(1), P.max(1)
        chosen = []
        for c in range(n_cls):
            members = np.where(lab == c)[0]
            members = members[np.argsort(-conf[members])]
            chosen += members[:int(len(members) * args.pseudo_frac)].tolist()
        tr = tr + [(paths[i], int(lab[i]), True) for i in chosen]
        print(f'pseudo-labels: {len(chosen)} / {len(paths)} test images added ({args.pseudo_frac:.0%} per class)')

    loader = DataLoader(ItemSet(tr, TrainTF()), batch_size=args.batch, shuffle=True,
                        drop_last=True, num_workers=6, persistent_workers=True)
    opt = torch.optim.AdamW([{'params': model.visual.parameters(), 'lr': args.lr},
                             {'params': [model.head], 'lr': args.lr * 20}], weight_decay=0.05)
    steps, warm = args.epochs * len(loader), len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
    ema, best = EMA(model), -1.0
    ckpt = ROOT / f'{tag}.pt'
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        for x, y in loader:
            x, y = x.to(dev).float(), y.to(dev)
            if np.random.rand() < 0.3:
                x, t = mix_batch(x, y, n_cls)
            else:
                t = torch.zeros(x.size(0), n_cls, device=dev).scatter_(1, y.view(-1, 1), 1.0)
            opt.zero_grad(set_to_none=True)
            soft_ce(model(x), t).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); ema.update(model)
        a = (probs(ema.m, va, dev).argmax(1) == yv).mean()
        mark = ''
        if a > best:
            best, mark = a, '  *'
            torch.save(ema.m.state_dict(), ckpt)
        print(f'ep {ep:>2}/{args.epochs} | ema val {a*100:5.2f}% | best {best*100:5.2f}% | {time.time()-t0:.0f}s{mark}')

    ft_state = torch.load(ckpt, map_location=dev)
    print('\nWiSE-FT interpolation (alpha=1 is plain fine-tuned):')
    results = {}
    for alpha in (0.5, 0.7, 1.0):
        model.load_state_dict(interpolate(zs_state, ft_state, alpha))
        Pv, Pt = report(f'alpha {alpha}', model)
        results[alpha] = (Pv, Pt)
        write_sub(Pt, paths, classes, f'sub_{tag}_a{int(alpha*10)}.csv')
        np.savez_compressed(ROOT / f'probs_{tag}_a{int(alpha*10)}.npz', test=Pt, val=Pv, yv=yv, scales=(RES,))
    print(f'\nsubmit sub_{tag}_zeroshot.csv, then sub_{tag}_a5.csv / a7 / a10')


if __name__ == '__main__':
    main()
