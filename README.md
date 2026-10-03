# 1st place — 0.98918

20-class image classification, Monash FIT3181 S2 2026. 1st of 156 teams on the private split,
up from 2nd on public. Six fine-tuned backbones, 99.68% validation, no external data.

## Two things decided it

Train and test were built by different processes.

**Test images are 64x64 JPEGs at quality 75**, LANCZOS-squashed to square. Training images are
full-resolution photos. Measuring that off the files and reproducing it in augmentation beat
every architecture change I tried.

**Twelve of the twenty folder names are wrong.** The ImageNet synset IDs are still in the
training filenames: `seals` are dugongs, `cakes` are trifles, `breads` are hot dogs, `ducks`
are geese. About 20% of the test set looks generated from the folder *word* instead, so for
that slice the label means something different than it does in training.

## The blend

| Member | Weight | Alone |
|---|---|---|
| PE-Core-L-336 | 2 | 0.98431 |
| MetaCLIP H-14 | 2 | 0.98420 |
| MetaCLIP H-14, two-sense head | 2 | 0.98484 |
| DINOv2-L | 3 | 0.98367 |
| EVA-02-L-448, ImageNet-supervised | 2 | 0.98175 |
| SigLIP 2 SO400M | 1 | 0.98046 |

Weights are each member's own leaderboard score. WiSE-FT alpha is tuned per family and the
spread is large — SigLIP 2 goes 0.96734 → 0.98046 between alpha 0.5 and 0.7.

Validation saturates at 944/947 while scores span 0.980–0.985, so it ranks nothing. Members
were scored individually on the board and changes accepted on a paired test: noise on a change
is sqrt(*d*) for *d* flipped predictions, so under ~15 images is a coin flip.

## Things that didn't work

| | |
|---|---|
| 1.9B model vs 0.4B of the same family | −52 |
| Distilling from the best ensemble instead of a weaker model | −19 |
| Sub-concept head without superclass reweighting | −12 |
| Swapping in a member 6 images better standalone | −3 |
| Confidence routing, robust aggregation, more members | 0 |

The members agree on 98% of images. Anything that recombines the same predictions moves tens
of images and scores nothing.

## Run it

```bash
pip install -r requirements.txt
python src/predict.py --probs probs --ids data/test_set/my_solution.csv --out submission.csv
```

Rebuilds the submission byte-for-byte from the stored arrays. `assignment_notebook.ipynb` has
the six training commands, about an hour each on one 4090. Full write-up in `report.pdf`.
Dataset not redistributed.
