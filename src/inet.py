"""Map ImageNet-1k probabilities to the 20 competition classes."""
import argparse, numpy as np, timm, torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import common as C
import kaggle_clip as kc
from kaggle_v4 import to_test_domain, write_sub

DOGS = list(range(151, 269)); SNAKES = list(range(52, 69)); FISH = [0, 1, 2, 3, 4, 5, 6] + list(range(389, 398))
BIRDS_OTHER = [i for i in range(10, 25) if i != 12] + list(range(80, 97)) + [100] + list(range(127, 147))
GROUP = {
    "birds": [12] + BIRDS_OTHER, "bottles": [440, 907, 898, 737, 720], "breads": [934, 930, 931, 932],
    "butterfiles": list(range(321, 327)), "cakes": [927], "cats": list(range(281, 286)), "chickens": [7, 8],
    "cows": [345], "dogs": DOGS, "ducks": [99, 97, 98], "elephants": [385, 386, 101], "fishes": FISH,
    "handguns": [597, 763], "horses": [339], "lions": [291], "lipsticks": [629], "seals": [149, 150],
    "snakes": SNAKES, "spiders": list(range(72, 78)), "vases": [883]}

class TF:
    def __init__(self, res, mean, std, crop=1.0):
        self.res, self.crop = res, crop; self.norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    def __call__(self, img, native):
        im = img if native else to_test_domain(img)
        if self.crop < 1.0:
            w, h = im.size; dw, dh = int(w * (1 - self.crop) / 2), int(h * (1 - self.crop) / 2); im = im.crop((dw, dh, w - dw, h - dh))
        return self.norm(im.resize((self.res, self.res), Image.BICUBIC))

@torch.no_grad()
def run(model, items, tf, dev, groups, batch=64):
    out = []
    for x, _ in DataLoader(kc.ItemSet(items, tf), batch_size=batch, num_workers=10):
        x = x.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = (torch.softmax(model(x).float(), 1) + torch.softmax(model(torch.flip(x, [3])).float(), 1)) / 2
        g = torch.stack([p[:, idx].sum(1) for idx in groups], 1)
        out.append(torch.cat([g / g.sum(1, keepdim=True), g.sum(1, keepdim=True)], 1).cpu())
    return torch.cat(out).numpy()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--res", type=int, default=448); ap.add_argument("--batch", type=int, default=64); args = ap.parse_args()
    dev = torch.device("cuda")
    model = timm.create_model(args.model, pretrained=True).to(dev).eval()
    cfg = timm.data.resolve_data_config(model=model); print(args.model, cfg["mean"], cfg["std"], flush=True)
    items, classes = C.train_items(); groups = [GROUP[c] for c in classes]
    order = C.split_order(len(items)); va = [items[i] for i in order]; yv = np.array([y for _, y, _ in va])
    tf = TF(args.res, cfg["mean"], cfg["std"])
    R = run(model, items, tf, dev, groups, args.batch); Pall, mass = R[:, :-1], R[:, -1]
    y = np.array([l for _, l, _ in items]); pred = Pall.argmax(1)
    print(f"\nALL 9,466 training images, class-mapped zero-shot: acc {(pred == y).mean():.4f}   mean in-group mass {mass.mean():.3f}")
    for c in range(len(classes)):
        m = y == c; print(f"  {classes[c]:12s} n={m.sum():4d}  acc {(pred[m] == y[m]).mean():.4f}  in-group mass {mass[m].mean():.3f}")
    Pv = Pall[order]; print(f"\nval split: acc {(Pv.argmax(1) == yv).mean():.4f}", flush=True)
    paths = C.test_paths(); Rt = run(model, [(p, 0, True) for p in paths], tf, dev, groups, args.batch); Pt = Rt[:, :-1]
    np.savez_compressed(C.ROOT / f"probs_{args.tag}.npz", test=Pt, val=Pv, yv=yv, scales=(args.res,), train_all=Pall, train_mass=mass, test_mass=Rt[:, -1])
    write_sub(Pt, paths, classes, f"sub_{args.tag}.csv")
    ch = np.load(C.ROOT / "probs_s_A_evaC.npz")["test"]; cp = ch.argmax(1)
    print(f"test: agree with champion {(Pt.argmax(1) == cp).mean():.4f}; mean in-group mass {Rt[:, -1].mean():.3f}; conf<0.5 {(Pt.max(1) < 0.5).mean():.3f}", flush=True)

if __name__ == "__main__":
    main()
