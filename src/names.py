"""Map class folders to ImageNet synset names."""
import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import common as C
import kaggle_clip as kc

SYNSET = {
    'n01443537': 'goldfish', 'n01532829': 'house finch', 'n01749939': 'green mamba',
    'n01855672': 'goose', 'n02074367': 'dugong', 'n02129165': 'lion',
    'n02823428': 'beer bottle', 'n03527444': 'holster', 'n03676483': 'lipstick',
    'n04522168': 'vase', 'n07613480': 'trifle', 'n07697537': 'hot dog',
}
EXTRA = {
    'snakes': ['snake', 'green mamba', 'green snake', 'mamba snake'],
    'birds': ['house finch', 'bird', 'finch', 'small brown bird'],
    'breads': ['hot dog', 'hotdog in a bun', 'bread', 'hot dog sausage in a bun'],
    'cakes': ['trifle', 'cake', 'layered dessert in a glass', 'dessert'],
    'seals': ['dugong', 'seal', 'manatee', 'sea cow'],
    'handguns': ['holster', 'handgun', 'gun in a holster', 'pistol'],
    'bottles': ['beer bottle', 'bottle', 'glass beer bottle'],
    'ducks': ['goose', 'duck', 'goose or duck'],
    'fishes': ['goldfish', 'fish', 'orange goldfish'],
    'lipsticks': ['lipstick', 'tube of lipstick'],
    'vases': ['vase', 'ceramic vase'],
    'lions': ['lion', 'male lion'],
}


def audit():
    """Return dominant synsets and class names."""
    items, classes = C.train_items()
    by = {c: Counter() for c in classes}
    for p, y, _ in items:
        m = re.match(r'(n\d{8})_', Path(p).name)
        by[classes[y]][m.group(1) if m else '-'] += 1
    out = {}
    for c in classes:
        syn, n = by[c].most_common(1)[0]
        out[c] = (syn, n / sum(by[c].values()))
    return out, classes


def text_head(names, classes, clip, tok):
    W = []
    with torch.no_grad():
        for c in classes:
            e = clip.encode_text(tok([t.format(names[c]) for t in kc.TEMPLATES]))
            e = e / e.norm(dim=-1, keepdim=True)
            W.append(e.mean(0))
    W = torch.stack(W)
    return (W / W.norm(dim=-1, keepdim=True)).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-search', action='store_true')
    args = ap.parse_args()
    import open_clip

    meta = np.load(C.CACHE / 'meta.npz')
    classes = [str(c) for c in meta['classes']]
    y = meta['y']
    order = C.split_order(len(y))          # kaggle_clip's val ORDER, not a mask -- see common.split_order
    Ev = np.load(C.CACHE / 'emb_train_matched_grey.npy').astype(np.float32)[order]
    Ete = np.load(C.CACHE / 'emb_test.npy').astype(np.float32)
    yv = y[order]

    prov, _ = audit()
    print('provenance from training filenames:')
    for c in classes:
        syn, share = prov[c]
        gloss = SYNSET.get(syn, '')
        print(f'  {c:14s} {syn if syn != "-" else "(no synset)":12s} {share:5.0%}  {gloss}')

    clip, _, _ = open_clip.create_model_and_transforms(C.ARCH, pretrained=C.PRETRAINED, force_image_size=C.RES)
    tok = open_clip.get_tokenizer(C.ARCH)
    names = dict(kc.NAMES)

    def val_acc(nm):
        return ((Ev @ text_head(nm, classes, clip, tok).T).argmax(1) == yv).mean()

    base = val_acc(names)
    print(f'\nfolder-name head: zero-shot val {base * 100:.2f}%')
    if not args.no_search:
        for c in classes:
            cands = EXTRA.get(c, [])
            syn_gloss = SYNSET.get(prov[c][0])
            if syn_gloss and syn_gloss not in cands:
                cands = [syn_gloss] + cands
            if not cands:
                continue
            best, bn = val_acc(names), names[c]
            for cand in cands:
                trial = {**names, c: cand}
                a = val_acc(trial)
                if a > best:
                    best, bn = a, cand
            if bn != names[c]:
                print(f'  {c:14s} "{names[c]}" -> "{bn}"   val {base * 100:.2f}% -> {best * 100:.2f}%')
                names[c] = bn
                base = best
    else:
        for c, g in ((c, SYNSET.get(prov[c][0])) for c in classes):
            if g:
                names[c] = g

    final = val_acc(names)
    W = text_head(names, classes, clip, tok)
    Pv = torch.softmax(torch.as_tensor(100.0 * Ev @ W.T), 1).numpy()
    Pt = torch.softmax(torch.as_tensor(100.0 * Ete @ W.T), 1).numpy()
    print(f'\ncorrected head: zero-shot val {final * 100:.2f}%  ({(final - (Ev @ text_head(kc.NAMES, classes, clip, tok).T).argmax(1).__eq__(yv).mean()) * 100:+.2f})')
    np.savez_compressed(C.ROOT / 'probs_zs_named.npz', test=Pt, val=Pv, yv=yv, scales=(C.RES,))
    paths = C.test_paths()
    disp = ['butterflies' if c == 'butterfiles' else c for c in classes]
    import pandas as pd
    pd.DataFrame({'ID': [C.test_id(p) for p in paths],
                  'Label': [disp[i] for i in Pt.argmax(1)]}).to_csv(C.ROOT / 'sub_zs_named.csv', index=False)
    (C.OUT / 'names.txt').write_text('\n'.join(f'{c}\t{names[c]}' for c in classes) + '\n')
    print(f'wrote probs_zs_named.npz, sub_zs_named.csv, {C.OUT / "names.txt"}')
    print('\nfinal names:  ' + ', '.join(f'{c}="{names[c]}"' for c in classes if names[c] != kc.NAMES[c]))


if __name__ == '__main__':
    main()
