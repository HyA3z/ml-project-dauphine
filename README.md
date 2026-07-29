# Predicting Organic Virality on TikTok

Machine learning project by **Franck Chen** and **Ziyang Chen**  
Université Paris Dauphine–PSL · April 2026

This project studies whether the organic virality of a TikTok video can be
predicted before publication. We focus on small and mid-sized creators
(3,000–30,000 followers), for whom performance is less dominated by an already
established audience.

The project combines:

- leakage-safe temporal feature engineering;
- gradient-boosting and stacking ensembles;
- niche-aware and speech-aware routing;
- thumbnail, caption, hashtag, and subtitle signals;
- a multimodal neural network; and
- binary and three-class formulations of creator-relative success.

See the complete [project report](ML_Project_report.pdf) and
[presentation slides](ML_Project_Slides.pdf).

## Research question

Raw view counts strongly depend on audience size. We therefore define an
**Explosion Score** that measures how far a video travels beyond its creator's
existing follower base:

```text
Explosion Score = video views / creator followers
target_log       = log(1 + Explosion Score)
```

The regression task predicts `target_log` from information available before a
video is posted:

1. the creator's performance over the previous 10 videos; and
2. the current video's metadata and multimodal content.

We also study a more actionable classification question: will a video
underperform, perform consistently, or exceed the creator's own historical
baseline?

## Dataset

Videos were collected with a hashtag-stratified strategy across six content
niches:

| Niche | Example content |
|---|---|
| Cooking | recipes, food, baking |
| Study | education, exams, study routines |
| Comedy | sketches, memes, humor |
| Fitness | workouts, gym, wellness |
| Tech | software, AI, gaming |
| Finance | investing, trading, business |

After filtering, the modeling dataset contains **466 creators** and **9,320
samples**. Each creator contributes 30 chronologically ordered videos:

- ranks 1–10 establish the initial historical window;
- ranks 11–26 form the training set: **7,456 samples**;
- ranks 27–28 form the validation set: **932 samples**;
- ranks 29–30 form the test set: **932 samples**.

This chronological split is used instead of a random split so that future
videos never influence past predictions.

### Feature families

- **Current metadata:** duration, posting hour, weekday, original-audio flag,
  caption, hashtags, thumbnail, and subtitles.
- **Historical performance:** median, 70th percentile, and 90th percentile of
  views over the previous 10 videos.
- **Normalized engagement:** historical like, comment, and share rates.
- **Temporal dynamics:** short- and long-term momentum, trend, acceleration,
  consistency, recovery, and streak features.
- **Text signals:** caption structure, hashtags, speech hooks, TF-IDF/LSA, and
  multilingual MiniLM embeddings.
- **Visual signals:** thumbnail brightness, contrast, and edge complexity.
- **Context signals:** creator target encoding, niche assignment, and
  creator–niche interactions.

All rolling and target-encoding features are shifted or computed out of fold to
prevent target leakage.

## Modeling pipeline

The experiments progress from simple baselines to specialized multimodal
architectures:

```text
Linear / KNN / Random Forest baselines
                 ↓
XGBoost + LightGBM + CatBoost
                 ↓
OOF stacking with a Ridge meta-learner
                 ↓
Temporal momentum + cross-features
                 ↓
Thumbnail, subtitle, and MiniLM features
                 ↓
Niche-aware / speech-aware expert routing
```

The final Mixture-of-Experts experiment routes each video to one of three
specialized stacks:

- **Comedy expert:** structural, momentum, and creator features;
- **Spoken expert:** structural features plus speech and MiniLM features;
- **Silent expert:** structural, momentum, and thumbnail features.

Each expert selects features locally and stacks XGBoost, LightGBM, CatBoost,
Random Forest, and KNN predictions with a Ridge meta-learner.

## Main results

The table below preserves the evaluation scope used in the report. Local and
spoken-only scores should not be compared directly with global test scores.

| Experiment | Evaluation scope | R² | MAE |
|---|---|---:|---:|
| Optimized XGBoost | Global | 0.2527 | 0.3461 |
| Stacking ensemble (v1) | Global | 0.3020 | 0.3030 |
| Temporal momentum (v2) | Global | 0.3260 | 0.2960 |
| Thumbnail features (v6) | Global | **0.3332** | 0.2968 |
| OOF target encoding + Optuna (v7) | Global | 0.3325 | **0.2922** |
| Niche-aware stack (v8) | Comedy only | **0.4036** | **0.2590** |
| LSA speech model (v9) | Global | 0.3176 | 0.2951 |
| MiniLM/BERT-SVD hybrid (v10) | Global | 0.3080 | 0.2955 |
| Dual pipeline (v11) | Spoken videos only | 0.3490 | 0.2953 |
| Creator + niche enrichment (v12) | Spoken videos only | 0.3693 | 0.3001 |
| Mixture of Experts (v13) | Comedy expert | 0.3895 | 0.3158 |
| Mixture of Experts (v13) | Global assembled | 0.2960 | 0.3300 |

### Deep learning

The multimodal MLP fuses 10 numerical features with a 384-dimensional
`paraphrase-multilingual-MiniLM-L12-v2` text embedding. A four-dimensional
text bottleneck prevents semantic features from overwhelming the structured
signals. The regression head uses 64 hidden units, dropout, AdamW, and Huber
loss.

| Model | R² | MAE |
|---|---:|---:|
| XGBoost reference baseline | 0.2420 | 0.3470 |
| Multimodal MLP | **0.3029** | **0.3208** |

The MLP improves explained variance by approximately 25% over this reference
baseline, but it does not exceed the best global tree ensemble.

### Classification

For classification, success is measured relative to each creator's historical
performance:

- **Binary:** viral if views exceed the creator's historical 70th percentile.
- **Three-class:** underperforming (below P50), consistent (P50–P70), or viral
  (above P70).

| Task | Best practical model | Accuracy | F1-macro | AUC-ROC |
|---|---|---:|---:|---:|
| Binary | Random Forest | 0.6599 | 0.5571 | 0.6309 |
| Three-class | Random Forest | 0.4861 | 0.4136 | 0.6195 |

KNN reaches 0.6384 accuracy on the three-class task, but mostly predicts the
majority class. Random Forest has lower nominal accuracy and substantially
better balanced discrimination.

## Key findings

1. **Global metadata models reach a plateau near R² ≈ 0.33.** Adding more
   tabular features does not automatically add signal.
2. **Creator momentum matters.** Short-term history and trajectory features
   consistently improve the baseline models.
3. **Multimodal features must be routed carefully.** Text embeddings help
   spoken videos, but forcing them into a dataset where roughly 65% of videos
   are silent degrades global tree models.
4. **Virality is niche-dependent.** Comedy is relatively momentum-driven and
   predictable, while niches such as Finance and Tech are more exposed to
   external trends.
5. **Creator-relative classification is more actionable than exact
   regression.** Predicting a breakout category is more robust than estimating
   the precise scale of a viral event.

## Repository structure

```text
.
├── ML_Project_report.pdf          # Full methodology and analysis
├── ML_Project_Slides.pdf          # Project presentation
├── dataset/
│   ├── scrape/                    # Apify scraping notebook and niche samples
│   ├── process/                   # Cleaning and subtitle-processing artifacts
│   ├── cleaned_data.csv           # Core 9,320-row modeling table
│   ├── cleaned_data_subtittles.csv
│   ├── cleaned_data_thumbnail.csv
│   ├── cleaned_data_ml.csv
│   └── dataset_engineered.csv
├── ml_method/
│   ├── train_simple.py            # Linear, Ridge, KNN, and RF baselines
│   ├── xgboost_tiktok_tuning.py   # Tuned XGBoost baseline
│   ├── gbdt_comparison_v1.py      # XGB / LGBM / CatBoost comparison
│   ├── stacking_v3_advanced.py    # Temporal stacking experiments
│   ├── stacking_v4_structural_fixes.py
│   ├── stacking_v5_nlp.py
│   ├── stacking_v6_cv.py
│   ├── stacking_v7_optuna.py
│   ├── stacking_v8_niche.py
│   ├── stacking_v9_speech_fixed.py
│   ├── stacking_v10_bert.py
│   ├── stacking_v11_dual.py
│   ├── stacking_v12.py
│   ├── stacking_v13_moe.py        # Final expert-routing experiment
│   └── classification.ipynb       # Binary and three-class experiments
└── dl_method/
    └── train.py                   # Multimodal MiniLM + numerical MLP
```

> The filename `cleaned_data_subtittles.csv` is intentionally shown with the
> spelling used in the repository because the experiment scripts reference it
> directly.

## Getting started

### 1. Create an environment

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/HyA3z/ml-project-dauphine.git
cd ml-project-dauphine

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install \
  numpy pandas scipy scikit-learn matplotlib seaborn pillow requests \
  textblob xgboost lightgbm catboost optuna \
  sentence-transformers torch jupyter
```

### 2. Run representative experiments

The scripts use paths relative to their working directory. From the repository
root, run the classical baselines with:

```bash
(cd ml_method && python train_simple.py)
```

Run advanced stacking experiments with `dataset/` as the working directory,
where their input CSV files are located:

```bash
(cd dataset && python ../ml_method/stacking_v7_optuna.py)
(cd dataset && python ../ml_method/stacking_v8_niche.py)
(cd dataset && python ../ml_method/stacking_v13_moe.py)
```

Run the multimodal neural network with:

```bash
(cd dl_method && python train.py)
```

Open the classification experiments with:

```bash
jupyter notebook ml_method/classification.ipynb
```

### Runtime notes

- The v6–v8 and v13 pipelines download video thumbnails from `coverUrl` and
  cache extracted image features locally.
- The v10–v13 and deep-learning pipelines may download the multilingual
  MiniLM model on first use and cache its embeddings.
- Optuna searches, repeated out-of-fold stacking, image downloads, and
  transformer encoding make the advanced experiments computationally
  expensive.
- Generated caches, plots, prediction CSVs, and model checkpoints are written
  to the directory from which the script is launched.

## Authors

- Franck Chen
- Ziyang Chen
