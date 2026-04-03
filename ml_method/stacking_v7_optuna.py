"""
Stacking v7 — Target Encoding OOF + Feature Selection + Optuna
================================================================
v6 → v7 : Three surgical levers against Feature Dilution

  [LEVIER A] OUT-OF-FOLD TARGET ENCODING (anti-leakage)
      creator_target_mean : creator's explosion_score average,
      calculated by STRICTLY excluding the target video via an OOF fold
      on the train set, then propagated to val/test with the global average.

  [LEVIER B] FEATURE SELECTION TOP-40
      Selection of the top 40 features via CatBoost importance
      on the full train set → reduction of colsample dilution.

  [LEVIER C] OPTUNA HYPERPARAMETER SEARCH (40 trials / model)
      Replacement of RandomizedSearch with an Optuna study using
      Hyperband pruning. Priority to low learning_rate + L1/L2 regularization.

  [KEPT] Strict chronological split  — Train 11-26 / Val 27-28 / Test 29-30
  [KEPT] Stacking L1 Ridge + L2 OOF
  [KEPT] CV features (PIL cache), NLP TextBlob, Momentum, Cross-features v4

Record to beat : R² = 0.3332 | MAE = 0.2968
Target         : R² ≥ 0.35
"""

import ast
import os
import io
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from PIL import Image, ImageFilter
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from textblob import TextBlob
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RECORD_R2  = 0.3332
RECORD_MAE = 0.2968
N_OPTUNA_TRIALS = 40   # Optuna trials per model
N_OOF_FOLDS     = 5    # folds for Target Encoding OOF
TOP_K_FEATURES  = 40   # features kept after selection

print("=" * 70)
print("  Stacking v7 — Target Encoding OOF + Feature Selection + Optuna")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# 1. LOADING & CREATOR_ID
# ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("cleaned_data_thumbnail.csv")
df = df.sort_values(["followers", "video_rank"]).reset_index(drop=True)
df["creator_id"] = df.groupby("followers").ngroup()

print(f"\n {df['creator_id'].nunique()} creators | {len(df):,} videos")
print(f"   coverUrl present : {df['coverUrl'].notna().sum():,} / {len(df):,}")

# ─────────────────────────────────────────────────────────────────────
# 2. STATIC FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────
def parse_hashtags(raw):
    try:
        tags = ast.literal_eval(raw) if isinstance(raw, str) else raw
        return [t["name"].lower() for t in tags if isinstance(t, dict)]
    except Exception:
        return []

df["_tags"]      = df["hashtag"].apply(parse_hashtags)
df["n_hashtags"] = df["_tags"].apply(len)
df["has_fyp"]    = df["_tags"].apply(lambda t: int(any("fyp"    in x for x in t)))
df["has_viral"]  = df["_tags"].apply(lambda t: int(any("viral"  in x for x in t)))
df["has_foryou"] = df["_tags"].apply(lambda t: int(any("foryou" in x for x in t)))
df.drop(columns=["_tags"], inplace=True)

df["caption_len"]            = df["caption"].fillna("").str.len()
df["has_emoji"]              = df["caption"].fillna("").apply(lambda s: int(any(ord(c) > 127 for c in s)))
df["has_question"]           = df["caption"].fillna("").str.contains(r"\?").astype(int)
df["has_exclamation"]        = df["caption"].fillna("").str.contains(r"!").astype(int)
df["viral_potential"]        = df["hist_p90_views"] / (df["hist_median_views"] + 1)
df["engagement_total_hist"]  = df["hist_like_rate"] + df["hist_comment_rate"] + df["hist_share_rate"]
df["is_peak_hour"]           = df["hour"].between(17, 22).astype(int)
df["follower_tier"]          = np.log1p(df["followers"])
df["views_efficiency_trend"] = df["hist_p70_views"] / (df["hist_median_views"] + 1)

# ─────────────────────────────────────────────────────────────────────
# [CV] COMPUTER VISION LAYER — PIL + requests (with cache)
# ─────────────────────────────────────────────────────────────────────
print("\n  Downloading & analyzing thumbnails (PIL + requests)...")

CV_CACHE_PATH = "cv_features_cache.csv"
SAVE_EVERY    = 100
t_cv = time.time()

def extract_image_features(url, timeout=5):
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None
        img = Image.open(io.BytesIO(resp.content)).convert("L")
        img_small = img.resize((64, 64))
        arr = np.array(img_small, dtype=np.float32)
        brightness = float(arr.mean())
        contrast   = float(arr.std())
        lap        = img_small.filter(ImageFilter.FIND_EDGES)
        lap_arr    = np.array(lap, dtype=np.float32)
        complexity = float(lap_arr.var())
        return brightness, contrast, complexity
    except Exception:
        return None

urls = df["coverUrl"].fillna("").tolist()
n    = len(urls)

if os.path.exists(CV_CACHE_PATH):
    cache_df = pd.read_csv(CV_CACHE_PATH, index_col="row_idx")
    done_idx = set(cache_df.index.tolist())
    print(f"    Cache found : {len(done_idx):,} images already processed out of {n:,}")
else:
    cache_df = pd.DataFrame(columns=["img_brightness", "img_contrast", "img_complexity"])
    cache_df.index.name = "row_idx"
    done_idx = set()
    print(f"    No cache — starting from scratch ({n:,} images)")

pending = [i for i in range(n) if i not in done_idx]
errors  = 0
batch   = {}
print(f"   Remaining images : {len(pending):,} | Saving every {SAVE_EVERY}\n")

for count, i in enumerate(pending):
    url  = urls[i]
    feat = None if not url else extract_image_features(url)
    batch[i] = feat if feat is not None else (np.nan, np.nan, np.nan)
    if feat is None:
        errors += 1

    if (count + 1) % SAVE_EVERY == 0 or count == len(pending) - 1:
        batch_df = pd.DataFrame.from_dict(
            batch, orient="index",
            columns=["img_brightness", "img_contrast", "img_complexity"]
        )
        batch_df.index.name = "row_idx"
        cache_df = pd.concat([cache_df, batch_df])
        cache_df.to_csv(CV_CACHE_PATH)
        batch = {}
        elapsed = time.time() - t_cv
        total_done = len(done_idx) + count + 1
        print(f"   [{total_done:>5}/{n}] errors={errors} | {elapsed:.0f}s")

cache_df = pd.read_csv(CV_CACHE_PATH, index_col="row_idx").sort_index()
for col in ["img_brightness", "img_contrast", "img_complexity"]:
    cache_df[col] = cache_df[col].fillna(cache_df[col].median())

df["img_brightness"] = cache_df["img_brightness"].values
df["img_contrast"]   = cache_df["img_contrast"].values
df["img_complexity"] = cache_df["img_complexity"].values
CV_COLS = ["img_brightness", "img_contrast", "img_complexity"]
print(f"\n   CV features ready in {time.time()-t_cv:.0f}s")

# ─────────────────────────────────────────────────────────────────────
# NLP LAYER (TextBlob)
# ─────────────────────────────────────────────────────────────────────
print("\n Calculating NLP features (TextBlob)...")
t_nlp = time.time()

def get_sentiment(text):
    try:
        blob = TextBlob(str(text))
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    except Exception:
        return 0.0, 0.0

captions   = df["caption"].fillna("")
sentiments = captions.apply(get_sentiment)
df["sentiment_polarity"]     = sentiments.apply(lambda x: x[0])
df["sentiment_subjectivity"] = sentiments.apply(lambda x: x[1])
df["emotional_intensity"]    = df["sentiment_polarity"].abs()
df["word_count"]             = captions.apply(lambda s: len(s.split()))
NLP_COLS = ["sentiment_polarity", "sentiment_subjectivity", "emotional_intensity", "word_count"]
print(f"   NLP features calculated in {time.time()-t_nlp:.1f}s")

# ─────────────────────────────────────────────────────────────────────
# 3. MOMENTUM
# ─────────────────────────────────────────────────────────────────────
print("\n Calculating Momentum...")

def rolling_slope(series, window=5):
    def slope(arr):
        if arr.isna().any() or len(arr) < 2:
            return np.nan
        return np.polyfit(np.arange(len(arr)), arr.values, 1)[0]
    return series.rolling(window, min_periods=2).apply(slope, raw=False)

def count_streak(series):
    def streak(arr):
        n = 0
        for i in range(len(arr) - 1, 0, -1):
            if arr[i] > arr[i-1]: n += 1
            else: break
        return n
    return series.rolling(5, min_periods=1).apply(streak, raw=True)

grp     = df.groupby("creator_id")["explosion_score"]
shifted = grp.shift(1)

df["momentum_3"]      = shifted.groupby(df["creator_id"]).transform(lambda s: s.rolling(3, min_periods=1).mean())
df["trend_slope"]     = shifted.groupby(df["creator_id"]).transform(lambda s: rolling_slope(s, window=5))
df["consistency"]     = shifted.groupby(df["creator_id"]).transform(lambda s: s.rolling(5, min_periods=2).std())
df["momentum_ratio"]  = df["momentum_3"] / (df["hist_median_views"] + 1)
df["trend_direction"] = np.sign(df["trend_slope"].fillna(0)).astype(int)
df["volatility_tier"] = df["consistency"] / (df["hist_median_views"] + 1)
df["momentum_7"]      = shifted.groupby(df["creator_id"]).transform(lambda s: s.rolling(7, min_periods=2).mean())
df["accel"]           = df["momentum_3"] - df["momentum_7"]
df["peak_score"]      = shifted.groupby(df["creator_id"]).transform(lambda s: s.rolling(5, min_periods=1).max())
recent_min            = shifted.groupby(df["creator_id"]).transform(lambda s: s.rolling(5, min_periods=1).min())
df["recovery"]        = (df["momentum_3"] - recent_min) / (df["peak_score"] - recent_min + 1)
df["streak_up"]       = shifted.groupby(df["creator_id"]).transform(lambda s: count_streak(s))
df["momentum_norm"]   = df["momentum_3"] / (df["hist_p90_views"] + 1)

MOMENTUM_COLS = [
    "momentum_3", "trend_slope", "consistency", "momentum_ratio",
    "trend_direction", "volatility_tier", "momentum_7", "accel",
    "peak_score", "recovery", "streak_up", "momentum_norm",
]
df[MOMENTUM_COLS] = df[MOMENTUM_COLS].fillna(df[MOMENTUM_COLS].median())

# ─────────────────────────────────────────────────────────────────────
# 4. CROSS-FEATURES (v4 + NLP + CV)
# ─────────────────────────────────────────────────────────────────────
print(" Calculating cross-features...")
df["mom3_x_tier"]      = df["momentum_3"]    * df["follower_tier"]
df["accel_x_viral"]    = df["accel"]          * df["viral_potential"]
df["recovery_x_hist"]  = df["recovery"]       * df["hist_median_views"]
df["streak_x_engage"]  = df["streak_up"]      * df["engagement_total_hist"]
df["peak_x_p90"]       = df["peak_score"]     / (df["hist_p90_views"] + 1)
df["mom7_x_consist"]   = df["momentum_7"]     / (df["consistency"] + 0.1)
df["trend_x_duration"] = df["trend_slope"]    * df["duration"]
df["norm_x_tier"]      = df["momentum_norm"]  * df["follower_tier"]
df["intensity_x_momentum"] = df["emotional_intensity"] * df["momentum_3"]
df["polarity_x_viral"]     = df["sentiment_polarity"]  * df["viral_potential"]
df["subj_x_tier"]          = df["sentiment_subjectivity"] * df["follower_tier"]
df["contrast_x_momentum"]  = df["img_contrast"]    * df["momentum_3"]
df["bright_x_viral"]       = df["img_brightness"]  * df["viral_potential"]
df["complex_x_tier"]       = df["img_complexity"]  * df["follower_tier"]

CV_CROSS_COLS  = ["contrast_x_momentum", "bright_x_viral", "complex_x_tier"]
NLP_CROSS_COLS = ["intensity_x_momentum", "polarity_x_viral", "subj_x_tier"]
CROSS_COLS = [
    "mom3_x_tier", "accel_x_viral", "recovery_x_hist", "streak_x_engage",
    "peak_x_p90", "mom7_x_consist", "trend_x_duration", "norm_x_tier",
] + NLP_CROSS_COLS + CV_CROSS_COLS

df[CROSS_COLS] = df[CROSS_COLS].replace([np.inf, -np.inf], np.nan).fillna(df[CROSS_COLS].median())

# ─────────────────────────────────────────────────────────────────────
# 5. FULL FEATURE SET (pre-selection)
# ─────────────────────────────────────────────────────────────────────
STATIC_FEATURES = [
    "followers", "duration", "hour", "weekday", "musicOriginal",
    "hist_median_views", "hist_p70_views", "hist_p90_views",
    "hist_like_rate", "hist_comment_rate", "hist_share_rate",
    "n_hashtags", "has_fyp", "has_viral", "has_foryou",
    "caption_len", "has_emoji", "has_question", "has_exclamation",
    "viral_potential", "engagement_total_hist", "is_peak_hour",
    "follower_tier", "views_efficiency_trend",
]
ALL_FEATURES = STATIC_FEATURES + MOMENTUM_COLS + CROSS_COLS + NLP_COLS + CV_COLS
TARGET = "target_log"

print(f"\n Raw feature set : {len(ALL_FEATURES)} features")

# ─────────────────────────────────────────────────────────────────────
# 6. STRICT CHRONOLOGICAL SPLIT
# ─────────────────────────────────────────────────────────────────────
train_mask = df["video_rank"].between(11, 26)
val_mask   = df["video_rank"].between(27, 28)
test_mask  = df["video_rank"].between(29, 30)

print(f"\n Train={train_mask.sum():,} | Val={val_mask.sum():,} | Test={test_mask.sum():,}")
print(f"   Record to beat : R²={RECORD_R2} | MAE={RECORD_MAE}\n")

# ─────────────────────────────────────────────────────────────────────
# [LEVIER A] OUT-OF-FOLD TARGET ENCODING (anti-leakage)
# ─────────────────────────────────────────────────────────────────────
print("─" * 70)
print("  [LEVIER A] Target Encoding Out-of-Fold (anti-leakage)")
print("─" * 70)

# Logic:
# 1. On the train set only, calculate the OOF target average for each
#    creator, excluding the target row from its own fold.
# 2. On val and test: use the global average per creator calculated
#    on ALL train data (no leakage since val/test are future).
# 3. Bayesian smoothing for creators with few videos.

SMOOTH_K = 5   # strength of Bayesian smoothing (pseudo-count)

train_df = df[train_mask].copy()
global_mean = train_df[TARGET].mean()

# OOF calculation on train
te_oof = np.zeros(len(train_df))
kf_te  = KFold(n_splits=N_OOF_FOLDS, shuffle=False)
train_idx_arr = np.arange(len(train_df))

for fold_i, (tr_idx, val_idx) in enumerate(kf_te.split(train_df)):
    # On the TR rows of this fold, calculate the average per creator
    tr_fold  = train_df.iloc[tr_idx]
    val_fold = train_df.iloc[val_idx]

    # Average + Bayesian smoothing: (n * mean_creator + k * global_mean) / (n + k)
    creator_stats = (
        tr_fold.groupby("creator_id")[TARGET]
        .agg(["mean", "count"])
        .rename(columns={"mean": "c_mean", "count": "c_count"})
    )
    creator_stats["te"] = (
        (creator_stats["c_mean"] * creator_stats["c_count"] + SMOOTH_K * global_mean)
        / (creator_stats["c_count"] + SMOOTH_K)
    )

    # Map to the validation rows of this fold
    for local_idx, row in zip(val_idx, val_fold.itertuples()):
        cid = row.creator_id
        if cid in creator_stats.index:
            te_oof[local_idx] = creator_stats.loc[cid, "te"]
        else:
            te_oof[local_idx] = global_mean

train_df["creator_target_mean"] = te_oof

# On val and test: global average per creator calculated on all train data
creator_full_stats = (
    train_df.groupby("creator_id")[TARGET]
    .agg(["mean", "count"])
    .rename(columns={"mean": "c_mean", "count": "c_count"})
)
creator_full_stats["te"] = (
    (creator_full_stats["c_mean"] * creator_full_stats["c_count"] + SMOOTH_K * global_mean)
    / (creator_full_stats["c_count"] + SMOOTH_K)
)

def map_te(subset_df):
    te_vals = subset_df["creator_id"].map(creator_full_stats["te"]).fillna(global_mean)
    return te_vals.values

val_df  = df[val_mask].copy()
test_df = df[test_mask].copy()
val_df["creator_target_mean"]  = map_te(val_df)
test_df["creator_target_mean"] = map_te(test_df)

# Rebuild df with the TE column on train/val/test
df.loc[train_mask, "creator_target_mean"] = te_oof
df.loc[val_mask,   "creator_target_mean"] = val_df["creator_target_mean"].values
df.loc[test_mask,  "creator_target_mean"] = test_df["creator_target_mean"].values

# Anti-leakage check: correlation must not be perfect on train
te_corr = np.corrcoef(df.loc[train_mask, "creator_target_mean"],
                      df.loc[train_mask, TARGET])[0, 1]
print(f"\n   creator_target_mean created (OOF {N_OOF_FOLDS}-fold, smoothing k={SMOOTH_K})")
print(f"  Pearson correlation with target (train) : {te_corr:.4f}")
print(f"  → If < 1.0 : no leakage ({' OK' if te_corr < 0.999 else ' LEAKAGE DETECTED'})")

ALL_FEATURES_V7 = ALL_FEATURES + ["creator_target_mean"]
print(f"\n  Feature set v7 (pre-selection) : {len(ALL_FEATURES_V7)} features")

# ─────────────────────────────────────────────────────────────────────
# [LEVIER B] FEATURE SELECTION TOP-{TOP_K_FEATURES}
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(f"  [LEVIER B] Feature Selection — Top {TOP_K_FEATURES} (CatBoost importance)")
print("─" * 70)

X_train_full = df.loc[train_mask, ALL_FEATURES_V7].astype(float)
y_train      = df.loc[train_mask, TARGET]
X_val_full   = df.loc[val_mask,   ALL_FEATURES_V7].astype(float)
y_val        = df.loc[val_mask,   TARGET]
X_test_full  = df.loc[test_mask,  ALL_FEATURES_V7].astype(float)
y_test       = df.loc[test_mask,  TARGET]

# Train a fast CatBoost for feature importance
print("\n  Fast CatBoost training for feature importance...")
cat_selector = CatBoostRegressor(
    iterations=3000, learning_rate=0.05, depth=6,
    early_stopping_rounds=100, random_state=42, verbose=0
)
cat_selector.fit(X_train_full, y_train, eval_set=(X_val_full, y_val))

fi_series = pd.Series(
    cat_selector.get_feature_importance(),
    index=ALL_FEATURES_V7
).sort_values(ascending=False)

SELECTED_FEATURES = fi_series.head(TOP_K_FEATURES).index.tolist()

print(f"\n  Top {TOP_K_FEATURES} selected features :")
for rank, (feat, imp) in enumerate(fi_series.head(TOP_K_FEATURES).items(), 1):
    tag = ""
    if feat == "creator_target_mean": tag = " ←  TARGET ENC"
    elif feat in CV_COLS:             tag = " ←   CV"
    elif feat in MOMENTUM_COLS:       tag = " ←  momentum"
    elif feat in NLP_COLS:            tag = " ←  NLP"
    elif feat in CV_CROSS_COLS:       tag = " ←   CV×"
    elif feat in CROSS_COLS:          tag = " ←  crossed"
    print(f"    #{rank:>2}  {feat:<35} {imp:>6.2f}%{tag}")

# Verify that creator_target_mean is included
te_rank = list(fi_series.index).index("creator_target_mean") + 1
te_imp  = fi_series["creator_target_mean"]
print(f"\n  → creator_target_mean : rank #{te_rank} | importance = {te_imp:.2f}%")
print(f"   Reduction : {len(ALL_FEATURES_V7)} → {len(SELECTED_FEATURES)} features")

X_train = df.loc[train_mask, SELECTED_FEATURES].astype(float)
X_val   = df.loc[val_mask,   SELECTED_FEATURES].astype(float)
X_test  = df.loc[test_mask,  SELECTED_FEATURES].astype(float)

# ─────────────────────────────────────────────────────────────────────
# [LEVIER C] OPTUNA — objective functions per model
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(f"  [LEVIER C] Optuna ({N_OPTUNA_TRIALS} trials × 3 models)")
print("─" * 70)

results = {}

# ── XGBoost ──────────────────────────────────────────────────────────
print(f"\n    XGBoost — {N_OPTUNA_TRIALS} Optuna trials")

def xgb_objective(trial):
    params = dict(
        max_depth        = trial.suggest_int   ("max_depth",        3, 7),
        learning_rate    = trial.suggest_float ("learning_rate",    0.002, 0.05,  log=True),
        n_estimators     = trial.suggest_int   ("n_estimators",     3000, 12000, step=1000),
        subsample        = trial.suggest_float ("subsample",        0.65, 0.95),
        colsample_bytree = trial.suggest_float ("colsample_bytree", 0.50, 0.90),
        min_child_weight = trial.suggest_int   ("min_child_weight", 2, 15),
        reg_alpha        = trial.suggest_float ("reg_alpha",        0.01, 5.0,   log=True),
        reg_lambda       = trial.suggest_float ("reg_lambda",       0.5,  10.0,  log=True),
        gamma            = trial.suggest_float ("gamma",            0.0,  0.5),
    )
    model = XGBRegressor(
        **params,
        tree_method="hist", early_stopping_rounds=200,
        eval_metric="mae", verbosity=0, random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    return -r2_score(y_val, preds)   # minimize → maximize R²

t0 = time.time()
xgb_study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=42),
    pruner=HyperbandPruner()
)
xgb_study.optimize(xgb_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
best_xgb  = xgb_study.best_params

print(f"  Best params XGB : {best_xgb}")
xgb_model = XGBRegressor(
    **best_xgb, tree_method="hist", early_stopping_rounds=200,
    eval_metric="mae", verbosity=0, random_state=42
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
y_pred_xgb = xgb_model.predict(X_test)
r2_xgb  = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
elapsed_xgb = time.time() - t0
print(f"  R²={r2_xgb:.4f} | MAE={mae_xgb:.4f} | {elapsed_xgb:.0f}s")
results["XGBoost"] = {"model": xgb_model, "r2": r2_xgb, "mae": mae_xgb,
                      "params": best_xgb, "time": elapsed_xgb}

# ── LightGBM ─────────────────────────────────────────────────────────
print(f"\n    LightGBM — {N_OPTUNA_TRIALS} Optuna trials")

def lgbm_objective(trial):
    params = dict(
        max_depth         = trial.suggest_int  ("max_depth",         3, 8),
        learning_rate     = trial.suggest_float("learning_rate",     0.002, 0.05, log=True),
        n_estimators      = trial.suggest_int  ("n_estimators",      3000, 12000, step=1000),
        num_leaves        = trial.suggest_int  ("num_leaves",        20, 127),
        subsample         = trial.suggest_float("subsample",         0.65, 0.95),
        colsample_bytree  = trial.suggest_float("colsample_bytree",  0.50, 0.90),
        min_child_samples = trial.suggest_int  ("min_child_samples", 10, 80),
        reg_alpha         = trial.suggest_float("reg_alpha",         0.01, 5.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",        0.5,  10.0, log=True),
    )
    model = LGBMRegressor(**params, random_state=42, verbose=-1)
    cb    = [lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)]
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=cb)
    preds = model.predict(X_val)
    return -r2_score(y_val, preds)

t0 = time.time()
lgbm_study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=42),
    pruner=HyperbandPruner()
)
lgbm_study.optimize(lgbm_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
best_lgbm  = lgbm_study.best_params

print(f"  Best params LGBM : {best_lgbm}")
lgbm_model = LGBMRegressor(**best_lgbm, random_state=42, verbose=-1)
cb_fit = [lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)]
lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=cb_fit)
y_pred_lgbm = lgbm_model.predict(X_test)
r2_lgbm  = r2_score(y_test, y_pred_lgbm)
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
elapsed_lgbm = time.time() - t0
print(f"  R²={r2_lgbm:.4f} | MAE={mae_lgbm:.4f} | {elapsed_lgbm:.0f}s")
results["LightGBM"] = {"model": lgbm_model, "r2": r2_lgbm, "mae": mae_lgbm,
                       "params": best_lgbm, "time": elapsed_lgbm}

# ── CatBoost ─────────────────────────────────────────────────────────
print(f"\n    CatBoost — {N_OPTUNA_TRIALS} Optuna trials")

def cat_objective(trial):
    params = dict(
        depth               = trial.suggest_int  ("depth",               4, 10),
        learning_rate       = trial.suggest_float("learning_rate",       0.002, 0.05, log=True),
        iterations          = trial.suggest_int  ("iterations",          3000, 12000, step=1000),
        l2_leaf_reg         = trial.suggest_float("l2_leaf_reg",         1.0, 20.0, log=True),
        subsample           = trial.suggest_float("subsample",           0.65, 0.95),
        colsample_bylevel   = trial.suggest_float("colsample_bylevel",   0.50, 0.90),
        min_data_in_leaf    = trial.suggest_int  ("min_data_in_leaf",    3, 30),
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0),
        random_strength     = trial.suggest_float("random_strength",     0.5, 5.0),
    )
    model = CatBoostRegressor(**params, early_stopping_rounds=200, random_state=42, verbose=0)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    preds = model.predict(X_val)
    return -r2_score(y_val, preds)

t0 = time.time()
cat_study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=42),
    pruner=HyperbandPruner()
)
cat_study.optimize(cat_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
best_cat  = cat_study.best_params

print(f"  Best params CAT : {best_cat}")
cat_model = CatBoostRegressor(**best_cat, early_stopping_rounds=200, random_state=42, verbose=0)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
y_pred_cat = cat_model.predict(X_test)
r2_cat  = r2_score(y_test, y_pred_cat)
mae_cat = mean_absolute_error(y_test, y_pred_cat)
elapsed_cat = time.time() - t0
print(f"  R²={r2_cat:.4f} | MAE={mae_cat:.4f} | {elapsed_cat:.0f}s")
results["CatBoost"] = {"model": cat_model, "r2": r2_cat, "mae": mae_cat,
                       "params": best_cat, "time": elapsed_cat}

# ─────────────────────────────────────────────────────────────────────
# STACKING L1 — Ridge meta-learner
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("    Stacking L1 — Ridge meta-learner")
print("─" * 70)
t0 = time.time()

xgb_s  = XGBRegressor(**best_xgb,  tree_method="hist", early_stopping_rounds=200,
                       eval_metric="mae", verbosity=0, random_state=42)
lgbm_s = LGBMRegressor(**best_lgbm, random_state=42, verbose=-1)
cat_s  = CatBoostRegressor(**best_cat, early_stopping_rounds=200, random_state=42, verbose=0)

xgb_s.fit(X_train, y_train,  eval_set=[(X_val, y_val)], verbose=False)
lgbm_s.fit(X_train, y_train, eval_set=[(X_val, y_val)],
           callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)])
cat_s.fit(X_train, y_train,  eval_set=(X_val, y_val))

p_xgb_val   = xgb_s.predict(X_val)
p_lgbm_val  = lgbm_s.predict(X_val)
p_cat_val   = cat_s.predict(X_val)
p_xgb_test  = xgb_s.predict(X_test)
p_lgbm_test = lgbm_s.predict(X_test)
p_cat_test  = cat_s.predict(X_test)

def build_meta_features(p_xgb, p_lgbm, p_cat):
    return np.column_stack([
        p_xgb, p_lgbm, p_cat,
        (p_xgb + p_lgbm + p_cat) / 3,
        np.abs(p_xgb  - p_cat),
        np.abs(p_lgbm - p_cat),
        np.abs(p_xgb  - p_lgbm),
        np.maximum(p_xgb, np.maximum(p_lgbm, p_cat)),
        np.minimum(p_xgb, np.minimum(p_lgbm, p_cat)),
    ])

meta_val_feats  = build_meta_features(p_xgb_val,  p_lgbm_val,  p_cat_val)
meta_test_feats = build_meta_features(p_xgb_test, p_lgbm_test, p_cat_test)

meta_ridge = Ridge(alpha=1.0)
meta_ridge.fit(meta_val_feats, y_val)
y_pred_ridge = meta_ridge.predict(meta_test_feats)
r2_ridge     = r2_score(y_test, y_pred_ridge)
mae_ridge    = mean_absolute_error(y_test, y_pred_ridge)
w_ridge      = meta_ridge.coef_[:3] / (meta_ridge.coef_[:3].sum() + 1e-9)
elapsed = time.time() - t0

print(f"  [Ridge L1] XGB={w_ridge[0]:.2%} | LGBM={w_ridge[1]:.2%} | CAT={w_ridge[2]:.2%}")
print(f"  R²={r2_ridge:.4f}  {'PASS' if r2_ridge > RECORD_R2 else 'FAIL'}  (Δ {r2_ridge - RECORD_R2:+.4f})")
print(f"  MAE={mae_ridge:.4f}  {'PASS' if mae_ridge < RECORD_MAE else 'FAIL'}")
results["Stacking-Ridge"] = {"r2": r2_ridge, "mae": mae_ridge, "time": elapsed}

# ─────────────────────────────────────────────────────────────────────
# STACKING L2 — OOF 5-fold
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("    Stacking L2 — OOF 5-fold")
print("─" * 70)
t0 = time.time()

kf = KFold(n_splits=5, shuffle=False)

oof_xgb  = cross_val_predict(
    XGBRegressor(**best_xgb,  tree_method="hist", verbosity=0, random_state=42),
    X_train, y_train, cv=kf
)
oof_lgbm = cross_val_predict(
    LGBMRegressor(**best_lgbm, random_state=42, verbose=-1),
    X_train, y_train, cv=kf
)
oof_cat  = cross_val_predict(
    CatBoostRegressor(**best_cat, random_state=42, verbose=0),
    X_train, y_train, cv=kf
)

meta_train_l2 = build_meta_features(oof_xgb, oof_lgbm, oof_cat)
meta_val_l2   = meta_val_feats
meta_test_l2  = meta_test_feats

# L2 meta-learners with Optuna
print("  Optuna L2 meta-learners...")

def meta_xgb_objective(trial):
    p = dict(
        max_depth     = trial.suggest_int  ("max_depth",     2, 5),
        learning_rate = trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
        n_estimators  = trial.suggest_int  ("n_estimators",  500, 5000, step=500),
        subsample     = trial.suggest_float("subsample",     0.7, 1.0),
        reg_alpha     = trial.suggest_float("reg_alpha",     0.01, 5.0, log=True),
        reg_lambda    = trial.suggest_float("reg_lambda",    0.5, 10.0, log=True),
    )
    m = XGBRegressor(**p, tree_method="hist", early_stopping_rounds=100,
                     eval_metric="mae", verbosity=0, random_state=77)
    m.fit(meta_train_l2, y_train, eval_set=[(meta_val_l2, y_val)], verbose=False)
    return -r2_score(y_val, m.predict(meta_val_l2))

meta_xgb_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=77))
meta_xgb_study.optimize(meta_xgb_objective, n_trials=20, show_progress_bar=False)
best_meta_xgb = meta_xgb_study.best_params

xgb_l2 = XGBRegressor(**best_meta_xgb, tree_method="hist", early_stopping_rounds=100,
                       eval_metric="mae", verbosity=0, random_state=77)
xgb_l2.fit(meta_train_l2, y_train, eval_set=[(meta_val_l2, y_val)], verbose=False)

cat_l2 = CatBoostRegressor(
    depth=4, learning_rate=0.01, iterations=3000,
    l2_leaf_reg=5, subsample=0.8,
    early_stopping_rounds=100, random_state=77, verbose=0
)
cat_l2.fit(meta_train_l2, y_train, eval_set=(meta_val_l2, y_val))

meta_l2_val  = np.column_stack([xgb_l2.predict(meta_val_l2),  cat_l2.predict(meta_val_l2)])
meta_l2_test = np.column_stack([xgb_l2.predict(meta_test_l2), cat_l2.predict(meta_test_l2)])

ridge_l2 = Ridge(alpha=1.0)
ridge_l2.fit(meta_l2_val, y_val)
y_pred_l2  = ridge_l2.predict(meta_l2_test)
r2_l2   = r2_score(y_test, y_pred_l2)
mae_l2  = mean_absolute_error(y_test, y_pred_l2)
elapsed = time.time() - t0

print(f"  R²={r2_l2:.4f}  {'PASS' if r2_l2 > RECORD_R2 else 'FAIL'}  (Δ {r2_l2 - RECORD_R2:+.4f})")
print(f"  MAE={mae_l2:.4f}  {'PASS' if mae_l2 < RECORD_MAE else 'FAIL'}")
results["Stacking-L2"] = {"r2": r2_l2, "mae": mae_l2, "time": elapsed}

# ─────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE — top 25 with feature layer
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print("    Feature Importance (Optimized CatBoost) — top 25")
print("─" * 70)

fi = pd.Series(cat_s.get_feature_importance(), index=SELECTED_FEATURES).sort_values(ascending=False)
print(f"\n  {'Feature':<35} {'Imp':>6}  Type")
print(f"  {'─'*62}")
for feat, imp in fi.head(25).items():
    if feat == "creator_target_mean": tag = " ←  TARGET ENC"
    elif feat in CV_COLS:             tag = " ←   CV"
    elif feat in CV_CROSS_COLS:       tag = " ←   CV×"
    elif feat in NLP_COLS:            tag = " ←  NLP"
    elif feat in NLP_CROSS_COLS:      tag = " ←  NLP×"
    elif feat in CROSS_COLS:          tag = " ← CROSSED"
    elif feat in MOMENTUM_COLS:       tag = " ← momentum"
    else:                             tag = ""
    print(f"  {feat:<35} {imp:>6.2f}%{tag}")

# ─────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL SUMMARY — v7 (Target Enc + Feature Sel + Optuna)")
print("=" * 70)
print(f"\n{'Model':<22} {'R²':>8} {'Δ R²':>9} {'MAE':>8} {'Δ MAE':>9}")
print("─" * 62)
for name, res in results.items():
    dr2  = res["r2"]  - RECORD_R2
    dmae = res["mae"] - RECORD_MAE
    icon = "* " if res["r2"] == max(r["r2"] for r in results.values()) else "  "
    print(f"{icon} {name:<20} {res['r2']:>8.4f} {dr2:>+9.4f} {res['mae']:>8.4f} {dmae:>+9.4f}")
print("─" * 62)
print(f"   {'Record v6':20} {RECORD_R2:>8.4f} {'':>9} {RECORD_MAE:>8.4f}")

best_name = max(results, key=lambda k: results[k]["r2"])
best_r2   = results[best_name]["r2"]
best_mae  = results[best_name]["mae"]
print(f"\n Best : {best_name}  (R²={best_r2:.4f} | MAE={best_mae:.4f})")
if best_r2 >= 0.35:
    print(" TARGET R² ≥ 0.35 ACHIEVED ! ")
elif best_r2 > RECORD_R2:
    print(f" New record ! Gain : +{best_r2 - RECORD_R2:.4f}")
else:
    print(" Persistent plateau — additional diagnostics required")

# ─────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle(
    "Stacking v7 — Target Enc OOF + Feature Selection + Optuna",
    fontsize=15, color="white", fontweight="bold", y=1.01
)

P = dict(
    ax_bg="#1a1a1a", grid="#2a2a2a", text="#f0f0f0",
    gold="#ffbe0b", record="#ff4d6d",
    mom="#00f5d4", cross="#ff9f1c", stat="#666666",
    nlp="#c77dff", nlp_cross="#e0aaff",
    cv="#ff6b6b", cv_cross="#ffa07a", te="#00d4ff"
)
COLORS = {
    "XGBoost": "#3a86ff", "LightGBM": "#06d6a0", "CatBoost": "#8338ec",
    "Stacking-Ridge": "#ffbe0b", "Stacking-L2": "#ef476f",
}

model_names = list(results.keys())
r2_vals     = [results[n]["r2"]  for n in model_names]
mae_vals    = [results[n]["mae"] for n in model_names]
bar_colors  = [COLORS.get(n, "#aaaaaa") for n in model_names]

for ax, vals, metric, higher_better in [
    (axes[0], r2_vals,  "R²",  True),
    (axes[1], mae_vals, "MAE", False),
]:
    ax.set_facecolor(P["ax_bg"])
    bars = ax.bar(model_names, vals, color=bar_colors, alpha=0.88, zorder=3, width=0.55)
    best_idx = (np.argmax(vals) if higher_better else np.argmin(vals))
    bars[best_idx].set_edgecolor(P["gold"])
    bars[best_idx].set_linewidth(2.5)
    bars[best_idx].set_alpha(1.0)
    record_val = RECORD_R2 if metric == "R²" else RECORD_MAE
    ax.axhline(record_val, color=P["record"], linewidth=1.5, linestyle="--", zorder=4,
               label=f"Record v6 ({record_val:.4f})")
    if metric == "R²":
        ax.axhline(0.35, color="#00b4d8", linewidth=1.0, linestyle=":", zorder=4, label="Target 0.35")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + (0.002 if higher_better else 0.001),
                f"{val:.4f}", ha="center", va="bottom", color=P["text"], fontsize=8, fontweight="bold")
    ax.set_title(metric, color=P["text"], fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Score", color=P["text"])
    ax.tick_params(colors=P["text"], labelsize=8)
    ax.yaxis.grid(True, color=P["grid"], linewidth=0.8, zorder=0)
    ax.legend(framealpha=0.2, labelcolor=P["text"], fontsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(P["grid"])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

# Feature importance top 20 colored by layer
ax = axes[2]
ax.set_facecolor(P["ax_bg"])
top20 = fi.head(20)

def feat_color(f):
    if f == "creator_target_mean": return P["te"]
    if f in CV_COLS:               return P["cv"]
    if f in CV_CROSS_COLS:         return P["cv_cross"]
    if f in NLP_COLS:              return P["nlp"]
    if f in NLP_CROSS_COLS:        return P["nlp_cross"]
    if f in CROSS_COLS:            return P["cross"]
    if f in MOMENTUM_COLS:         return P["mom"]
    return P["stat"]

fi_colors = [feat_color(f) for f in top20.index]
ax.barh(range(len(top20)), top20.values, color=fi_colors, alpha=0.88, zorder=3)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.index, fontsize=7.5, color=P["text"])
ax.invert_yaxis()
ax.set_title(f"Feature Importance (CatBoost) — Top {TOP_K_FEATURES} selected",
             color=P["text"], fontsize=11, fontweight="bold", pad=10)
ax.tick_params(colors=P["text"], labelsize=8)
ax.xaxis.grid(True, color=P["grid"], linewidth=0.8, zorder=0)
for sp in ax.spines.values(): sp.set_edgecolor(P["grid"])

# Recalculating layer totals for the legend
sel = set(SELECTED_FEATURES)
cv_t       = fi[[c for c in CV_COLS if c in sel]].sum()
cv_cx_t    = fi[[c for c in CV_CROSS_COLS if c in sel]].sum()
nlp_t      = fi[[c for c in NLP_COLS if c in sel]].sum()
nlp_cx_t   = fi[[c for c in NLP_CROSS_COLS if c in sel]].sum()
cross_t    = fi[[c for c in CROSS_COLS if c in sel and c not in NLP_CROSS_COLS and c not in CV_CROSS_COLS]].sum()
mom_t      = fi[[c for c in MOMENTUM_COLS if c in sel]].sum()
stat_t     = fi[[c for c in STATIC_FEATURES if c in sel]].sum()
te_t       = fi.get("creator_target_mean", 0)

legend_handles = [
    mpatches.Patch(color=P["te"],       label=f"Target Enc OOF ({te_t:.1f}%)  ← NEW"),
    mpatches.Patch(color=P["cv"],       label=f"CV PIL        ({cv_t:.1f}%)"),
    mpatches.Patch(color=P["cv_cross"], label=f"CV × Cross    ({cv_cx_t:.1f}%)"),
    mpatches.Patch(color=P["nlp"],      label=f"NLP TextBlob  ({nlp_t:.1f}%)"),
    mpatches.Patch(color=P["cross"],    label=f"Crossed       ({cross_t:.1f}%)"),
    mpatches.Patch(color=P["mom"],      label=f"Momentum      ({mom_t:.1f}%)"),
    mpatches.Patch(color=P["stat"],     label=f"Static        ({stat_t:.1f}%)"),
]
ax.legend(handles=legend_handles, framealpha=0.2, labelcolor=P["text"], fontsize=7, loc="lower right")

plt.tight_layout()
out_path = "stacking_v7_optuna.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n Chart saved → {out_path}")
plt.close()

print("\n Stacking v7 completed.\n")