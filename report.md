# FIT3181/5215 Deep Learning — S2 2026 — Kaggle Competition Report

**Student ID:** 33970076  **Student name:** Harvey Houlahan  **Display name on Kaggle:** Harvey Houlahan

**Private leaderboard 0.98918 — 1st of 156 teams, from 176 entrants.** Public 0.98548. Six-member ensemble; 99.68% validation (944/947)

---

## ABSTRACT

Training images are mostly full-resolution photos; test images are 64x64 JPEGs at quality 75. I
measured that pipeline and used it in training. I also found that 12 folder names do not match their ImageNet synsets: `seals` contain dugongs, `cakes`, trifles, and `breads`, hot dogs. The final submission blends six fine-tuned backbones. Validation plateaued at 944/947, so I selected members by leaderboard score. The entry placed 1st of 156 teams on the private split at 0.98918, from 2nd on the public split.

## SET UP

Python 3.12, PyTorch 2.11 (CUDA 12.8), open_clip 3.3, timm 1.0.29, torchvision 0.26, NumPy 2.5, pandas 3.0, Pillow. Trained on 4× RTX 4090 (48 GB). Pretrained backbone weights are downloaded from Hugging Face by open_clip/timm at first run. No external training images are used.

`train.py` trains CLIP members. `predict.py` rebuilds the submission from stored probabilities.

## DATASET

9,466 labelled images across 20 classes, split 90/10 by a seeded permutation (seed 1234) into 8,519 training and 947 validation images; the test set is 11,681 images.

Test images are 64x64 JPEGs at quality 75 with 4:2:0 chroma, made with LANCZOS resizing.
Non-native training images are resized and re-encoded at quality 65-85 before augmentation.
The native 64x64 fishes and lions files are left unchanged.

Normalization uses each backbone's own CLIP mean/std. Augmentation is RandomResizedCrop, horizontal flip, TrivialAugmentWide, random greyscale (12%, matching the test set's 9.3% greyscale rate), and random erasing.

## METHODS

### Model architecture

The submission averages six probability arrays. Each pretrained vision tower uses a cosine classifier initialized from class-name text embeddings.

| Member | Backbone | Weight |
|---|---|---|
| PE-Core-L-336 | Meta Perception Encoder, ViT-L/14 @ 336 | 2 |
| MetaCLIP H-14 | ViT/H-14 | 2 |
| MetaCLIP H-14 (Two-sense head) | ViT-H/14, expanded class names | 2 |
| DINOv2-L | ViT-L/14, self-supervised | 3 |
| EVA-02-L-448 | ImageNet-1k supervised, 20 classes by synset grouping | 2 |
| SigLIP 2 SO400M | sigmoid-loss contrastive | 1 |

Weights follow standalone leaderboard score.

### Techniques

- **Domain matching:** resize and JPEG-encode training images like test images.
- **Text head:** initialise class weights from text embeddings.
- **WiSE-FT:** interpolate zero-shot and fine-tuned weights per model family.
- **Self-training:** add the top 90% confident predictions per class as pseudo-labels.
- **Two-sense head:** use folder and synset names for mismatched classes. All members use mixup/cutmix, EMA, label smoothing, and flip-based test-time augmentation.

## EXPERIMENT AND RESULTS

### Parameter setting

| Parameter | Setting | Basis |
|---|---|---|
| Learning rate | 4e-6 (5e-6 for ConvNeXt) | 1e-5 diverged; 1e-6 underfit |
| Epochs | 5-9 by backbone | 9 vs 7 epochs: -4 images |
| Batch size | 24 | largest that fits at 336–448px |
| WiSE-FT alpha | 0.7 PE-Core / MetaCLIP / SigLIP 2; 1.0 DINOv2 | swept per family |
| Pseudo-label fraction | 0.90 per class | |
| Optimiser | AdamW, wd 0.05, head LR ×20 | |
| Schedule | 1 epoch linear warmup, then cosine | |

WiSE-FT alpha is family-specific. SigLIP 2 improved from 0.96734 at 0.5 to 0.98046 at 0.7;
PE-Core scored 0.98431 at 0.7 and 0.98303 at 1.0.

### Ledger

Public score after each accepted change. One image is 1/9,374 = 0.000107.

| # | Change | Public | Δ |
|---|---|---|---|
| 1 | Trained CNN at 64px | 0.59605 | — |
| 2 | Used a deeper CNN with multi-scale TTA | 0.70821 | +1,051 img |
| 3 | Switched to ResNet34 and matched the test domain | 0.80096 | +869 img |
| 4 | Added zero-shot CLIP ViT-B/16 text head | 0.85720 | +527 img |
| 5 | Fine-tuned with WiSE-FT | 0.93319 | +712 img |
| 6 | Used DataComp-XL, alpha=0.5, and self-training | 0.96296 | +279 img |
| 7 | Replaced ViT-B/16 with ViT-L/14 | 0.97673 | +129 img |
| 8 | Blended ViT-H/14 and ViT-L/14 | 0.97897 | +21 img |
| 9 | Added ConvNeXt-L | 0.98014 | +11 img |
| 10 | Added EVA-02-L with three seeds | 0.98303 | +27 img |
| 11 | Added SigLIP at 1/11 weight | 0.98367 | +6 img |
| 12 | Kept cross-teacher members beside originals | 0.98388 | +2 img |
| 13 | Rebuilt with four current backbones | 0.98495 | +10 img |
| 14 | Weighted members by measured score | 0.98527 | +3 img |
| 15 | Added MetaCLIP at full weight | 0.98537 | +1 img |
| **16** | **Kept both versions of two members (submitted)** | **0.98548** | **+1 img** |

Measured and rejected:

| Change | Public | Δ |
|---|---|---|
| Used hand labels for training | 0.97459 | -20 img |
| Swapped in a standalone-better member | 0.98505 | -3 img |
| Used a sub-concept head without superclass reweighting | 0.98356 | -12 img |
| Distilled from the strongest blend | 0.97844 | -19 img |
| Scaled to a 1.9B same-family model | 0.97491 | -52 img |

### Loss and training accuracy

| Measure | Result |
|---|---|
| Training loss | Smoothed cross-entropy; epoch loss was not retained. |
| Training accuracy | Not used: augmentation and pseudo-labels make it non-comparable. |
| Validation (947 held-out images) | **99.68%** (944/947) |
| Public leaderboard test split (9,374 images) | **0.98548** (2nd) |
| Private leaderboard test split (2,307 images) | **0.98918** (1st) |

Validation did not rank candidates: nine of 11 configurations scored 943/947 while public scores ranged from 0.98014 to 0.98388. I used paired leaderboard comparisons; changes below about 15 images were treated as noise. The final gain was 15 images across 47 changed predictions: z = 2.20, p = 0.028. The split confirmed the caution: 12 images behind on the public 80%, 1 image ahead on the private 20% — 25 errors against the runner-up's 26.

## DISCUSSION AND CONCLUSION

The largest gains came from observing of the data: test-image processing and incorrect folder names proved more significant than model iterations; as validation became saturated, leaderboard comparisons became the rubric. About one third of remaining errors contain no target class.

## REFERENCES AND RELATED WORKS

1. Wortsman et al., *Robust fine-tuning of zero-shot models* (WiSE-FT), CVPR 2022.
2. Novack et al., *CHiLS: Zero-Shot Image Classification with Hierarchical Label Sets*, ICML 2023.
3. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021.
4. Zhai et al., *Sigmoid Loss for Language Image Pre-Training* (SigLIP), ICCV 2023.
5. Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, TMLR 2024.
6. Bolya et al., *Perception Encoder*, Meta AI, 2025.
7. Xu et al., *Demystifying CLIP Data* (MetaCLIP), ICLR 2024.
8. Zhang et al., *mixup: Beyond Empirical Risk Minimization*, ICLR 2018.
9. Yun et al., *CutMix*, ICCV 2019.
10. Touvron et al., *Fixing the train-test resolution discrepancy* (FixRes), NeurIPS 2019.

## SUBMISSION DECLARATIONS

- [x] I have provided accurate information in every detail of this report and in any attached documents.
- [x] I understand that deductions for Assignment 1 will occur if I submit this report after the due date or provide any missing or incorrect information.
- [x] I have submitted this report alongside my trained model. Evaluating my trained model on the test set yields the same (or almost exactly the same) accuracy as displayed on the Kaggle Leaderboard.
- [x] I have compressed the files into a .zip extension and renamed it to 33970076_assignment01_report.zip.
