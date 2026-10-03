"""Shared paths, data helpers, and leaderboard scores."""
import glob
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))

from kaggle_v4 import DATA_DIR, TEST_DIR, NATIVE, SEED, to_test_domain  # noqa: E402

CACHE, OUT = PKG / 'cache', PKG / 'out'
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

ARCH, PRETRAINED, RES = 'ViT-B-16', 'datacomp_xl_s13b_b90k', 224
FINAL_PROBS = ROOT / 'probs_clip_vitb16_pl_a5.npz'

# Recorded submitted scores.
LB = {
    'probs_v4_resnet34':          0.801,
    'probs_zs_ViTB16quickgelu_openai_160': 0.857,
    'probs_clip_vitb16_pl_zs':    0.931,
    'probs_clip_a7':              0.933,
    'probs_clip_vitb16_pl_a10':   0.95624,
    'probs_clip_wise_a4':         0.96051,
    'probs_clip_vitb16_pl_a7':    0.96061,
    'probs_clip_vitb16_pl_a5':    0.96296,
    'probs_blend2_tau0':          0.96104,
    'probs_blend5_tau0':          0.95453,
    'probs_blend3_tau0':          0.93649,
}
LB_N_HALF = 11681 / 2

# WiSE-FT alpha and probability file.
LADDER = [
    (0.0, 'probs_clip_vitb16_pl_zs'),
    (0.2, 'probs_clip_wise_a2'),
    (0.3, 'probs_clip_wise_a3'),
    (0.4, 'probs_clip_wise_a4'),
    (0.5, 'probs_clip_vitb16_pl_a5'),
    (0.7, 'probs_clip_vitb16_pl_a7'),
    (1.0, 'probs_clip_vitb16_pl_a10'),
]

TEST_GREY_FRAC, TRAIN_GREY_FRAC = 0.093, 0.021


def train_items():
    """Return training items and class names."""
    from torchvision import datasets
    base = datasets.ImageFolder(DATA_DIR)
    items = [(p, y, Image.open(p).size == (NATIVE, NATIVE)) for p, y in base.samples]
    return items, base.classes


def split_order(n):
    """Return validation indices in saved-array order."""
    import torch
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randperm(n, generator=g).tolist()
    return np.array(idx[int(0.9 * n):])


def split_mask(n):
    """Return a validation-row mask."""
    m = np.zeros(n, bool)
    m[split_order(n)] = True
    return m


def test_paths():
    return sorted(glob.glob(str(TEST_DIR / '*_animal.jpeg')))


def test_id(path):
    return int(Path(path).name.split('_', 1)[0])


def _stable_unit(path):
    """Return a stable value in [0, 1] for a path."""
    return int(hashlib.md5(str(path).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _grey(img):
    return ImageOps.grayscale(img).convert('RGB')


def _is_grey(img, tol=6):
    a = np.asarray(img.convert('RGB')).astype(int)
    return (np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()) < tol


def lvl_raw(img, path):
    return img


def lvl_bilinear64(img, path):
    return img.resize((NATIVE, NATIVE), Image.BILINEAR)


def lvl_lanczos64(img, path):
    return img.resize((NATIVE, NATIVE), Image.LANCZOS)


def lvl_matched(img, path):
    return to_test_domain(img)


def lvl_matched_grey(img, path):
    extra = (TEST_GREY_FRAC - TRAIN_GREY_FRAC) / (1 - TRAIN_GREY_FRAC)
    if _stable_unit(path) < extra and not _is_grey(img):
        img = _grey(img)
    return to_test_domain(img)


LEVELS = {
    'raw': lvl_raw,
    'bilinear64': lvl_bilinear64,
    'lanczos64': lvl_lanczos64,
    'matched': lvl_matched,
    'matched_grey': lvl_matched_grey,
}

STYLES = [
    'a photo of a {}.',
    'a black and white photo of a {}.',
    'a pencil sketch of a {}.',
    'a line drawing of a {}.',
    'a watercolor painting of a {}.',
    'an oil painting of a {}.',
    'a cartoon of a {}.',
    'a 3d render of a {}.',
    'clip art of a {}.',
    'a sculpture of a {}.',
]
PHOTO_STYLES = 2


def load_probs(name):
    """Load test, validation, and label arrays."""
    f = ROOT / f'{name}.npz'
    if not f.exists():
        f = ROOT / 'FIT3181_Assignment1_Kaggle_33970076' / 'probs' / f'{name}.npz'
    r = np.load(f)
    return {'test': r['test'], 'val': r['val'], 'yv': r['yv']}


def all_probs_names():
    """List probability files with validation arrays."""
    out = []
    for p in sorted(ROOT.glob('probs_*.npz')):
        if p.stem != 'probs_blend' and 'val' in np.load(p).files:
            out.append(p.stem)
    return out


def lb_of(sub_stem):
    """Return a recorded score for a submission stem."""
    return LB.get('probs_' + sub_stem) or LB.get('probs_clip_' + sub_stem)
