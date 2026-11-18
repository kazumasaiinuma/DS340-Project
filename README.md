# AI Photo Culling (DS340 Final Project)

Alex Bischof, Kazumasa Iinuma — Fall 2026

Pick the best photo from a burst of near-duplicate shots. Each photo is scored on quality
features (subject sharpness, eye state, expression) combined with CLIP image embeddings, and a
pairwise ranker trained on human preferences predicts which shot people prefer.

## Data

[Princeton/Adobe Photo Triage Benchmark](https://pixl.cs.princeton.edu/pubs/Chang_2016_ATF/index.php)
(Chang et al., SIGGRAPH 2016), downloaded from the Kaggle mirror
[`ericwolter/triage`](https://www.kaggle.com/datasets/ericwolter/triage).

- Only `train_val/` is used: the official test set has no released labels.
- `train_pairlist.txt` / `val_pairlist.txt` columns:
  `series  photo1  photo2  pref_ratio(photo1 over photo2)  rank1  rank2` (rank 1 = best in series).
- Images: `train_val_imgs/{series:06d}-{photo:02d}.JPG`, resized to fit 800x800.
- 12,988 images, 12,075 training pairs, 483 validation pairs.
- Our test set: 15% of training **series** held out (`random_state=0`), split by series, never by
  pair — 1,905 pairs across 684 series, mean series size 2.7 photos.

Images are not committed (13.65 GB). The notebook downloads them.

## Running it

Everything lives in `notebooks/AIPhotoCulling.ipynb`, written for Google Colab.

| Section | What it does | Needs |
|---|---|---|
| 0 | Resume a later session from cached outputs in Drive (or uploaded files) | — |
| 1 | Kaggle login, download, extract `train_val/` | Kaggle API key |
| 2 | Per-photo features: face/global sharpness, MediaPipe blink and smile | CPU, ~8 min |
| 3 | Sharpest-photo baseline | — |
| 4 | CLIP ViT-B/32 embeddings | GPU, ~7 min |
| 5 | Pairwise logistic-regression ranker | — |
| 6 | Ablations, McNemar tests, face-pair subset | — |
| 7 | Top-1 accuracy and the retention curve | — |
| 8 | Sharpness on its own, and whether CLIP already encodes it | — |
| 9 | Per-feature explanation of a single decision | — |
| 10 | Feature standardisation check | — |

Sections 2 and 4 cache their output (`features_v1.csv`, `clip_b32.npy`, `clip_keys.csv`), so later
sessions can run section 0 and skip the 13.65 GB download entirely.

## Results

Pairwise accuracy on the held-out test split (1,905 pairs). "Confident" = pairs where at least
70% of annotators agreed.

| Method | All pairs | Confident |
|---|---|---|
| Sharpest photo (Laplacian variance, no training) | 0.597 | 0.637 |
| Hand features only | 0.599 | 0.646 |
| CLIP ViT-B/32 only | 0.641 | 0.718 |
| CLIP + hand features | 0.652 | 0.726 |
| Chang et al., best prior baseline (different split) | 0.57 | — |
| Chang et al., best model (fine-tuned Siamese VGG) | — | 0.73 |

**Hand features do not significantly improve the ranker overall** — McNemar's test gives 125
corrections against 103 regressions, p = 0.164. They do help on the 255 test pairs where both
photos contain a face (0.635 to 0.686, p = 0.047), but no single feature reaches significance on
its own there: sharpness 0.671 (p = 0.136), blink 0.667, expression 0.639.

**Only 13.8% of images have a detected face**, which caps what the face-derived features can
contribute on this dataset.

**Sharpness is partly redundant with CLIP; eye state is not.** Ridge regression from the 32 CLIP
dimensions predicts log-sharpness at R² = 0.42 on held-out photos, but eye closure at only
R² = 0.08 on photos containing a face. (The R² = 0.36 for blink across all photos is an artifact
of filling blink with 0 where no face is found — that regression mostly detects faces, which CLIP
does at R² = 0.59.)

### Picking one winner per group

Top-1 accuracy over the 684 test series is **0.575** (chance 0.373; sharpest-photo rule 0.520),
rising to 0.666 on the 377 series where annotators agreed decisively.

Retention on the 283 series with three or more photos:

| Rule | Photos discarded | Favourite retained |
|---|---|---|
| Keep 1 | 73% | 0.435 |
| Keep 2 | 45% | 0.777 |
| Keep 3 | 18% | 0.933 |

Holding 95% retention permits discarding only about 18% of photos, so the tool ranks photos and
moves rejects to a review folder rather than deleting anything. A score-margin cut-off performs
no better than fixed top-k at matched discard rates, because the margin is a monotone transform
of the model's predicted win probability.

Standardising every feature by its training standard deviation before fitting changes test
accuracy by at most 0.4 points (CLIP only 0.641 to 0.637; CLIP plus hand features 0.652 to
0.655), so the reported results use unstandardised features.

## Status

Done: feature extraction, CLIP embeddings, the pairwise ranker, ablations with significance
tests, top-1 and retention analysis, the redundancy check.

Next: a Siamese ResNet-18 fine-tuned on preference pairs; a CEW-trained eye-state
model, gated on first benchmarking MediaPipe's blink score against CEW's labels; burst grouping
from CLIP embeddings and capture time; and a separate, people-heavy set of longer bursts shot and
labelled by us, since Photo Triage series average 2.7 photos and are mostly face-free.

Cut from the proposal: per-user personalization, time-based grouping, the web interface, and a
FER-2013 expression model (MediaPipe's expression score adds nothing even on face pairs, and
FER-2013's 48x48 grayscale images would degrade our face crops).

## Citation

Huiwen Chang, Fisher Yu, Jue Wang, Douglas Ashley, Adam Finkelstein.
"Automatic Triage for a Photo Series." ACM Transactions on Graphics (Proc. SIGGRAPH) 35(4), 2016.
