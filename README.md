# AI Photo Culling (DS340 Final Project)

Alex Bischof, Kazumasa Iinuma — Fall 2026

Pick the best photo from a burst of near-duplicate shots. Each photo is scored on quality
features (subject sharpness, eye state, expression), combined with CLIP image embeddings,
and a pairwise ranker trained on human preferences predicts which shot people prefer.

## Data

[Princeton/Adobe Photo Triage Benchmark](https://pixl.cs.princeton.edu/pubs/Chang_2016_ATF/index.php)
(Chang et al., SIGGRAPH 2016), downloaded from the Kaggle mirror
[`ericwolter/triage`](https://www.kaggle.com/datasets/ericwolter/triage).

- Only `train_val/` is used: the official test set has no released labels.
- `train_pairlist.txt` / `val_pairlist.txt` columns:
  `series  photo1  photo2  pref_ratio(photo1 over photo2)  rank1  rank2` (rank 1 = best in series).
- Images: `train_val_imgs/{series:06d}-{photo:02d}.JPG`, resized to fit 800×800.
- 12,988 images, 12,075 training pairs, 483 validation pairs.
- Our test set: 15% of training **series** held out (`random_state=0`), split by series, never by pair.

Images are not committed (13.65 GB). Download them with `scripts/download_data.sh`.

## Setup (Google Colab)

```bash
pip install -r requirements.txt
export KAGGLE_USERNAME=your_username   # set KAGGLE_KEY / KAGGLE_API_TOKEN via getpass, never commit it
bash scripts/download_data.sh          # -> data/train_val/
```

Outputs (features, embeddings) go to `$OUT_DIR` (default `outputs/`). In Colab, point it at Drive:

```bash
export DATA_DIR=/content/data/train_val
export OUT_DIR=/content/drive/MyDrive/ds340
```

## Pipeline

| Step | Script | Output | Status |
|---|---|---|---|
| 1. Per-photo features: face/global sharpness, MediaPipe blink & smile | `src/features.py` | `features_v1.csv` | done |
| 2. Sharpest-photo baseline | `src/eval_sharpness.py` | printed accuracy | done |
| 3. CLIP ViT-B/32 embeddings (GPU) | `src/clip_embed.py` | `clip_b32.npy`, `clip_keys.csv` | not run yet |
| 4. Pairwise logistic-regression ranker | `src/ranker.py` | printed accuracy | not run yet |

## Results so far

Pairwise accuracy on the 483 validation pairs. "Confident" = pairs where at least 70% of
people agreed (ratio ≥ 0.7 or ≤ 0.3).

| Method | All pairs | Confident pairs (n=292) |
|---|---|---|
| Sharpest photo (face-region Laplacian variance, else whole image) | 0.553 | 0.592 |
| Best prior baseline reported by Chang et al. (different split) | 0.57 | — |

Only 13.8% of images have a detected face, so face features (eye state, smile) affect a
small share of pairs on this dataset. Eye-state models are evaluated on face pairs only.

## Citation

Huiwen Chang, Fisher Yu, Jue Wang, Douglas Ashley, Adam Finkelstein.
"Automatic Triage for a Photo Series." ACM Transactions on Graphics (Proc. SIGGRAPH) 35(4), 2016.
