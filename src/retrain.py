"""Train a CLIP member and emit WiSE-FT probability files."""
import argparse
import contextlib
import copy
import math
import random
import time

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps, ImageChops, ImageEnhance
from torch.utils.data import DataLoader, Dataset

import common as C
import kaggle_clip as kc
from kaggle_v4 import soft_ce, write_sub
from names import SYNSET, audit


WORK = 144
MIN_STRUCT = 0.45
MIN_INK = 0.04


def _work(img):
    return img.resize((WORK, WORK), Image.LANCZOS)


def _edges(img, thresh, grow=1):
    e = ImageOps.grayscale(img).filter(ImageFilter.GaussianBlur(0.6)).filter(ImageFilter.FIND_EDGES)
    if grow:
        e = e.filter(ImageFilter.MaxFilter(3))
    return e.point(lambda v: 0 if v > thresh else 255)


def _sketch(img):
    """Create a pencil-style image."""
    g = ImageOps.grayscale(_work(img))
    inv = ImageOps.invert(g).filter(ImageFilter.GaussianBlur(radius=random.uniform(2.0, 3.5)))
    a, b = np.asarray(g, np.float32), np.asarray(inv, np.float32)
    out = np.clip(a * 255.0 / np.maximum(255.0 - b, 1.0), 0, 255).astype(np.uint8)
    out = ImageOps.autocontrast(Image.fromarray(out), cutoff=1)
    return out.point(lambda v: int(255 * (v / 255) ** 1.6)).convert('RGB')


def _coloured_pencil(img):
    """Create a coloured-pencil image."""
    w = _work(img)
    wash = Image.blend(ImageEnhance.Color(w).enhance(1.3), Image.new('RGB', w.size, 'white'), 0.55)
    return ImageChops.multiply(wash, _sketch(img))


def _poster(img):
    """Create a cartoon-style image."""
    w = _work(img).filter(ImageFilter.MedianFilter(size=7))
    w = ImageOps.posterize(w, random.choice([3, 4]))
    w = ImageEnhance.Color(w).enhance(random.uniform(1.4, 1.8))
    w = ImageEnhance.Brightness(w).enhance(1.12)
    outline = Image.blend(_edges(_work(img), 60).convert('RGB'), Image.new('RGB', w.size, 'white'), 0.45)
    return ImageChops.multiply(w, outline)


def _painting(img):
    """Create a painterly image."""
    w = _work(img).resize((96, 96), Image.LANCZOS).filter(ImageFilter.ModeFilter(size=random.choice([5, 7])))
    w = ImageEnhance.Color(w).enhance(random.uniform(1.3, 1.8))
    w = ImageEnhance.Contrast(w).enhance(random.uniform(1.1, 1.4))
    return w.resize((WORK, WORK), Image.BICUBIC).filter(ImageFilter.SMOOTH)


def _flat_render(img):
    """Create a render-style image."""
    w = _work(img).filter(ImageFilter.GaussianBlur(1.0))
    w = ImageOps.posterize(w, 4)
    bloom = w.filter(ImageFilter.GaussianBlur(4)).point(lambda v: int(v * 0.45))
    return ImageChops.add(ImageEnhance.Contrast(w).enhance(1.15), bloom.convert('RGB'))


STYLE_FNS = [_sketch, _coloured_pencil, _poster, _painting, _flat_render]


def _struct(a, b):
    """Return grayscale structural correlation."""
    fa = np.asarray(ImageOps.grayscale(a.resize((64, 64), Image.LANCZOS)), np.float32).ravel()
    fb = np.asarray(ImageOps.grayscale(b.resize((64, 64), Image.LANCZOS)), np.float32).ravel()
    if fa.std() < 1e-3 or fb.std() < 1e-3:
        return 0.0
    return float(np.corrcoef(fa, fb)[0, 1])


def _styled(img, tries=3):
    """Return a valid styled view or the original."""
    for _ in range(tries):
        fn = random.choice(STYLE_FNS)
        out = fn(img)
        small = out.resize((64, 64), Image.LANCZOS)
        ink = (np.asarray(ImageOps.grayscale(small), np.float32) < 235).mean()
        if abs(_struct(img, out)) >= MIN_STRUCT and ink >= MIN_INK:
            return out
    return img




class StyledTrainTF(kc.TrainTF):
    """Training transform with optional style augmentation."""

    def __init__(self, p_style):
        super().__init__()
        self.p = p_style

    def __call__(self, img, native):
        if self.p > 0 and random.random() < self.p:
            img = _styled(img)
            if random.random() < 0.3:
                img = ImageOps.grayscale(img).convert('RGB')
        return super().__call__(img, native)


class SoftSet(Dataset):
    """Dataset with one-hot or soft targets."""

    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, t, nat = self.items[i]
        return self.tf(Image.open(p).convert('RGB'), nat), torch.as_tensor(t, dtype=torch.float32)


def mix_soft(x, t):
    """Apply mixup or cutmix to soft targets."""
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
    return x, lam * t + (1 - lam) * t[perm]


class CropEvalTF(kc.EvalTF):
    """Evaluation transform with a centre crop."""

    def __init__(self, crop):
        super().__init__()
        self.crop = crop

    def __call__(self, img, native):
        if self.crop < 1.0:
            w, h = img.size
            dw, dh = int(w * (1 - self.crop) / 2), int(h * (1 - self.crop) / 2)
            img = img.crop((dw, dh, w - dw, h - dh))
        return super().__call__(img, native)


@torch.no_grad()
def probs_tta(model, items, dev, batch=128):
    """Return the five-view prediction average."""
    model.eval()
    outs = []
    for crop in (1.0, 0.92, 0.85):
        acc = []
        for x, _ in DataLoader(kc.ItemSet(items, CropEvalTF(crop)), batch_size=batch, num_workers=4):
            x = x.to(dev).float()
            p = torch.softmax(model(x), 1)
            if crop != 0.85:
                p = (p + torch.softmax(model(torch.flip(x, [3])), 1)) / 2
            acc.append(p.cpu())
        outs.append(torch.cat(acc))
    return ((outs[0] * 2 + outs[1] * 2 + outs[2]) / 5).numpy()


def text_head(clip, tok, classes, names):
    """Average prompt embeddings for each class."""
    W = []
    with torch.no_grad():
        for c in classes:
            ns = names[c] if isinstance(names[c], list) else [names[c]]
            e = clip.encode_text(tok([t.format(n) for n in ns for t in kc.TEMPLATES]))
            e = e / e.norm(dim=-1, keepdim=True)
            W.append(e.mean(0))
    W = torch.stack(W)
    return W / W.norm(dim=-1, keepdim=True)


def name_sets(classes):
    prov, _ = audit()
    folder = dict(kc.NAMES)
    syn = {c: SYNSET.get(prov[c][0], folder[c]) for c in classes}
    syn.update({'seals': 'manatee', 'snakes': 'mamba snake', 'vases': 'ceramic vase'})
    union = {c: sorted({folder[c], syn[c]}) for c in classes}
    return folder, syn, union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--pseudo', default='probs_clip_vitb16_pl_a5.npz')
    ap.add_argument('--pseudo-frac', type=float, default=0.75)
    ap.add_argument('--init', default='union', choices=['union', 'folder', 'synset', 'twosense'])
    ap.add_argument('--anchor', default='folder', choices=['union', 'folder', 'synset', 'twosense'])
    ap.add_argument('--human-labels', default=None)
    ap.add_argument('--human-holdout', type=float, default=0.25,
                    help='human-label evaluation holdout')
    ap.add_argument('--human-repeat', type=int, default=2,
                    help='human-label repeat count')
    ap.add_argument('--soup-with', default='clip_vitb16_pl.pt')
    ap.add_argument('--alphas', nargs='*', type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
    ap.add_argument('--tag', default='r2')
    ap.add_argument('--seed', type=int, default=C.SEED)
    ap.add_argument('--style-aug', type=float, default=0.0)
    ap.add_argument('--soft', action='store_true')
    ap.add_argument('--soft-temp', type=float, default=0.5)
    ap.add_argument('--tta', action='store_true')
    ap.add_argument('--model', default='ViT-B-16')
    ap.add_argument('--pretrained', default=None, help='open_clip tag')
    ap.add_argument('--res', type=int, default=None)
    ap.add_argument('--amp', action='store_true', help='use bf16 autocast')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--extra-dir', default=None, help='extra ImageFolder data')
    ap.add_argument('--extra-repeat', type=int, default=1)
    ap.add_argument('--chils', action='store_true', help='use a CHiLS group-max head')
    args = ap.parse_args()

    kc.RES = C.RES
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available()
                       else 'mps' if torch.backends.mps.is_available() else 'cpu')

    items, classes = C.train_items()
    K = len(classes)
    order = C.split_order(len(items))
    val_set = set(order.tolist())
    tr = [it for i, it in enumerate(items) if i not in val_set]
    va = [items[i] for i in order]
    yv = np.array([y for _, y, _ in va])
    paths = C.test_paths()
    test_items = [(p, 0, True) for p in paths]
    counts = np.bincount([y for _, y, _ in items], minlength=K).astype(float)
    prior = counts / counts.sum() * len(paths)

    arch = args.model
    pre = args.pretrained or C.PRETRAINED
    if args.res: kc.RES = args.res
    clip, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pre, force_image_size=kc.RES)
    tok = open_clip.get_tokenizer(arch)
    folder, syn, union = name_sets(classes)
    from twosense import names_for as _twosense_names
    heads = {'folder': folder, 'synset': syn, 'union': union,
             'twosense': _twosense_names(classes)}
    if args.chils:
        from chils import SubConceptClf, build_subconcept_head
        Wsub, gid, flat = build_subconcept_head(clip, tok, classes, heads[args.init], kc.TEMPLATES)
        # Use folder embeddings as the WiSE anchor.
        Wf = text_head(clip, tok, classes, heads[args.anchor])
        Wanc = Wf[gid].clone()
        model = SubConceptClf(clip.visual, Wsub, gid, len(classes)).to(dev)
        del clip
        zs_state = copy.deepcopy(model.state_dict())
        zs_state['head'] = Wanc.to(dev).clone()
        print(f'CHiLS group-max head: {Wsub.shape[0]} sub-concepts over {len(classes)} classes', flush=True)
    else:
        W_init = text_head(clip, tok, classes, heads[args.init])
        W_anchor = text_head(clip, tok, classes, heads[args.anchor])
        model = kc.Clf(clip.visual, W_init).to(dev)
        del clip
        zs_state = copy.deepcopy(model.state_dict())
        zs_state['head'] = W_anchor.to(dev).clone()
    print(f'{args.tag} | {arch} | device {dev} | init={args.init} anchor={args.anchor} | {len(tr)} train / {len(va)} val', flush=True)

    onehot = lambda c: np.eye(K, dtype=np.float32)[c]
    tr = [(p, onehot(y), nat) for p, y, nat in tr]
    if args.extra_dir:
        from pathlib import Path as _P
        _ed = _P(args.extra_dir)
        _ci = {c: i for i, c in enumerate(classes)}
        _rows = []
        for _d in sorted(_ed.iterdir()):
            if not _d.is_dir() or _d.name not in _ci:
                continue
            for _f in sorted(_d.glob("*")):
                _rows.append((str(_f), onehot(_ci[_d.name]), False))
        tr = tr + _rows * args.extra_repeat
        print(f"extra data: {len(_rows)} images x{args.extra_repeat} from {_ed}", flush=True)
    P = np.load(C.ROOT / args.pseudo)['test']
    lab, conf = P.argmax(1), P.max(1)
    human = {}
    if args.human_labels:
        raw = pd.read_csv(args.human_labels)
        raw['key'] = raw['key'].astype(str) if 'key' in raw.columns else raw['ID'].astype(str)
        raw = raw[~raw['key'].str.endswith('#r') & (raw.label != 'unsure')]
        disp = {('butterflies' if c == 'butterfiles' else c): i for i, c in enumerate(classes)}
        every = {int(r.ID): disp[r.label] for r in raw.itertuples()}
        # Keep a deterministic evaluation holdout.
        keys = np.array(sorted(every))
        rng = np.random.default_rng(args.seed)
        held = set(keys[rng.random(len(keys)) < args.human_holdout].tolist())
        human = {i: c for i, c in every.items() if i not in held}
        (C.OUT / f'trained_ids_{args.tag}.txt').write_text('\n'.join(str(i) for i in sorted(human)) + '\n')
        print(f'human labels: {len(every)} usable -> {len(human)} for training x{args.human_repeat}, '
              f'{len(held)} held out for evaluation (frontier/out/trained_ids_{args.tag}.txt)', flush=True)

    id_of = {C.test_id(p): i for i, p in enumerate(paths)}
    human_rows = {id_of[i] for i in human if i in id_of}
    extra = []
    if args.soft:
        Q = P ** (1.0 / args.soft_temp)
        Q /= Q.sum(1, keepdims=True)
        extra = [(paths[i], Q[i].astype(np.float32), True) for i in range(len(paths)) if i not in human_rows]
        print(f'soft pseudo-labels: all {len(extra)} test images, temperature {args.soft_temp}')
    else:
        chosen = []
        for c in range(K):
            members = np.where(lab == c)[0]
            members = np.array([m for m in members if m not in human_rows], dtype=int)
            members = members[np.argsort(-conf[members])]
            chosen += members[:int(len(members) * args.pseudo_frac)].tolist()
        extra = [(paths[i], onehot(int(lab[i])), True) for i in chosen]
        print(f'pseudo-labels: {len(chosen)} / {len(paths)} ({args.pseudo_frac:.0%} per class)')
    extra += [(paths[id_of[i]], onehot(y), True) for i, y in human.items() if i in id_of] * args.human_repeat
    tr = tr + extra
    if args.style_aug > 0:
        print(f'style augmentation: p={args.style_aug} over {len(STYLE_FNS)} filters, structure-gated')

    loader = DataLoader(SoftSet(tr, StyledTrainTF(args.style_aug)), batch_size=args.batch, shuffle=True,
                        drop_last=True, num_workers=args.workers, persistent_workers=True)
    opt = torch.optim.AdamW([{'params': model.visual.parameters(), 'lr': args.lr},
                             {'params': [model.head], 'lr': args.lr * 20}], weight_decay=0.05)
    steps, warm = args.epochs * len(loader), len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    ema = kc.EMA(model)
    ckpt_last, ckpt_best = C.ROOT / f'clip_{args.tag}_last.pt', C.ROOT / f'clip_{args.tag}_best.pt'
    best = -1.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        for x, t in loader:
            x, t = x.to(dev).float(), t.to(dev)
            if np.random.rand() < 0.3:
                x, t = mix_soft(x, t)
            opt.zero_grad(set_to_none=True)
            if args.amp:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = soft_ce(model(x), t)
            else:
                loss = soft_ce(model(x), t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); ema.update(model)
        with (torch.autocast('cuda', dtype=torch.bfloat16) if args.amp else contextlib.nullcontext()):
            a = (kc.probs(ema.m, va, dev).argmax(1) == yv).mean()
        if a > best:
            best = a
        torch.save(ema.m.state_dict(), ckpt_last)
        print(f'ep {ep:>2}/{args.epochs} | ema val {a * 100:5.2f}% | best {best * 100:5.2f}% | {time.time() - t0:.0f}s', flush=True)

    def emit(name, state):
        model.load_state_dict(state)
        pf = probs_tta if args.tta else kc.probs
        ctx = torch.autocast('cuda', dtype=torch.bfloat16) if args.amp else contextlib.nullcontext()
        with ctx:
            Pv, Pt = pf(model, va, dev), pf(model, test_items, dev)
        np.savez_compressed(C.ROOT / f'probs_{name}.npz', test=Pt, val=Pv, yv=yv, scales=(C.RES,))
        write_sub(Pt, paths, classes, f'sub_{name}.csv')
        print(f'  {name:22s} val {(Pv.argmax(1) == yv).mean() * 100:5.2f}%  balance-err {kc.balance(Pt, prior):.3f}  '
              f'conf<0.5 {(Pt.max(1) < 0.5).mean() * 100:4.1f}%', flush=True)

    ft = torch.load(ckpt_last, map_location=dev)
    print('\nWiSE-FT ladder, new model (last-epoch EMA):')
    for a in args.alphas:
        emit(f'{args.tag}_a{int(round(a * 10)):02d}', kc.interpolate(zs_state, ft, a))
    old = C.ROOT / args.soup_with
    if old.exists() and arch == 'ViT-B-16':
        ft_old = torch.load(old, map_location=dev)
        soup = {k: (ft[k] + ft_old[k]) / 2 if ft[k].dtype.is_floating_point else ft[k] for k in ft}
        print('\nWiSE-FT ladder, soup(new, old):')
        for a in args.alphas:
            emit(f'{args.tag}soup_a{int(round(a * 10)):02d}', kc.interpolate(zs_state, soup, a))
    print('\ndone. score everything with:  python -m frontier.eval')


if __name__ == '__main__':
    main()
