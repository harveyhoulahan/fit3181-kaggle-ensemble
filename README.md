# 20-class image classification — 1st place, 0.98918

Winning entry for the FIT3181/5215 Deep Learning Kaggle competition, Monash University, S2 2025.
**1st of 156 teams** on the private leaderboard at **0.98918**, from 2nd on the public split.

A six-member ensemble of fine-tuned vision-language backbones. 99.68% validation accuracy
(944/947). No external training images.

## The problem

20 classes, 9,466 labelled training images, 11,681 test images. The interesting part is that
the two sets were built by different processes, and most of the work was figuring out how.

**Test images are 64x64 JPEGs at quality 75**, squashed to square with a LANCZOS filter.
Training images are mostly full-resolution photographs. Measuring that pipeline from the files
and reproducing it in the training augmentation was worth more than any architectural change.

**Twelve of the twenty folder names do not describe their contents.** Reading the ImageNet
synset IDs off the training filenames shows `seals` holds dugongs, `cakes` holds trifles,
`breads` holds hot dogs, `ducks` holds geese. Around 20% of the test set appears to be
generated from the folder *word* rather than the synset, so for a large slice of the test set
the label means something different than it does in training.

## The ensemble

A weighted average of six probability arrays. Each member is a pretrained vision tower with a
cosine classifier initialised from class-name text embeddings.

| Member | Backbone | Weight | Standalone |
|---|---|---|---|
| PE-Core-L-336 | Meta Perception Encoder ViT-L/14 | 2 | 0.98431 |
| MetaCLIP H-14 | ViT-H/14 | 2 | 0.98420 |
| MetaCLIP H-14, two-sense head | ViT-H/14, expanded class names | 2 | 0.98484 |
| DINOv2-L | ViT-L/14, self-supervised | 3 | 0.98367 |
| EVA-02-L-448 | ImageNet-1k supervised, grouped to 20 classes | 2 | 0.98175 |
| SigLIP 2 SO400M | sigmoid-loss contrastive | 1 | 0.98046 |

Weights follow each member's own measured leaderboard score.

## What mattered

**Domain matching.** Training images are pushed through the measured test pipeline before
augmentation. The two natively-64px classes are left alone, because degrading them twice
collapses them.

**WiSE-FT, tuned per family.** Final weights interpolate between fine-tuned and zero-shot.
The optimal interpolation is family-specific and the difference is large: SigLIP 2 scores
0.96734 at alpha 0.5 and 0.98046 at alpha 0.7, a 136-image swing from one hyper-parameter.
PE-Core moves the other way.

**Two-sense head.** For the twelve mismatched folders the classifier is initialised from both
senses at once, so `seals` carries sea lion *and* dugong. Worth +6 images on MetaCLIP.

**Selecting on the leaderboard, not validation.** Validation saturates at 944/947 while
leaderboard scores span 0.980 to 0.985, so it cannot rank candidates. Every member was scored
individually and changes were accepted using a paired test: for *d* changed predictions the
noise on a change is sqrt(*d*), so a move under ~15 images is not distinguishable from chance.

## What did not work

Kept because the negative results were more informative than the positive ones.

| Attempt | Result |
|---|---|
| Scale up: 1.9B model against 0.4B of the same family | −52 images |
| Distil from the strongest ensemble instead of a weaker model | −19 images |
| Sub-concept head without superclass reweighting | −12 images |
| Swap in a member that is 6 images better standalone | −3 images |
| Confidence routing, robust aggregation, extra members | 0 |

The last row is the general shape of it: the members agree on about 98% of images, so anything
that recombines the same predictions moves tens of images and scores nothing. Only replacing
backbones or adding information the models lacked ever helped.

## Reproduce

```bash
pip install -r requirements.txt
python src/predict.py --probs probs --ids data/test_set/my_solution.csv --out submission.csv
```

This rebuilds `submission.csv` byte-for-byte from the stored member arrays. To retrain from
scratch, see `assignment_notebook.ipynb` for the six commands; roughly one hour per member on
one RTX 4090. Pretrained weights download from Hugging Face on first run.

## Layout

| Path | What |
|---|---|
| `report.pdf` | Full write-up |
| `assignment_notebook.ipynb` | Executable record: training commands, trace, rebuild |
| `src/train.py` | Main trainer: domain matching, text head, WiSE-FT, self-training |
| `src/dinotrain.py` | DINOv2 variant; no text tower, so the head is a fitted linear probe |
| `src/inettrain.py` | ImageNet-supervised variant; 1000 logits grouped to 20 classes |
| `src/twosense.py` | The two-sense class names |
| `src/predict.py` | Rebuilds the submission from `probs/` |
| `probs/` | Six member probability arrays over the test set |

The dataset is not redistributed here.
