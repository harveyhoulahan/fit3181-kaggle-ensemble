"""Build a submission from stored member probability arrays."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MEMBERS = {
    "f1peL_a07":    2,
    "k2meta_a07":   2,
    "v3metaT_a07":  2,
    "d1dino_a10":   3,
    "g1eva448_a07": 2,
    "f2sig2_a07":   1,
}
CLASSES = ["birds", "bottles", "breads", "butterfiles", "cakes", "cats", "chickens", "cows",
           "dogs", "ducks", "elephants", "fishes", "handguns", "horses", "lions", "lipsticks",
           "seals", "snakes", "spiders", "vases"]
DISPLAY = {"butterfiles": "butterflies"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="probs", help="directory holding probs_<member>.npz")
    ap.add_argument("--ids", default="submission.csv", help="CSV used for ID order")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    root = Path(args.probs)
    total = sum(MEMBERS.values())
    blend = None
    for name, w in MEMBERS.items():
        p = np.load(root / f"probs_{name}.npz")["test"]
        blend = p * w if blend is None else blend + p * w
        print(f"  {name:16s} weight {w}  shape {p.shape}")
    blend /= total

    ids = pd.read_csv(args.ids)["ID"].tolist()
    assert len(ids) == blend.shape[0], f"{len(ids)} ids vs {blend.shape[0]} rows"
    labels = [DISPLAY.get(CLASSES[i], CLASSES[i]) for i in blend.argmax(1)]
    pd.DataFrame({"ID": ids, "Label": labels}).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(ids)} rows, {len(set(labels))} classes")


if __name__ == "__main__":
    main()
