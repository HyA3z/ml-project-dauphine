"""
Stacking v6 — Computer Vision Layer
=====================================
v5 → v6 : Injection of the Computer Vision layer (PIL + requests) on TikTok thumbnails.

  [NEW 1] COMPUTER VISION FEATURES (PIL + requests)
      img_brightness  : average thumbnail brightness [0-255]
      img_contrast    : pixel standard deviation (contrast measure)
      img_complexity  : local entropy via gradient variance (Laplacian variance)
      → Fallback : ensemble median in case of download error

  [KEPT] NLP Layer (TextBlob) — sentiment_polarity, subjectivity, emotional_intensity, word_count
  [KEPT] Momentum Layer       — momentum_3, trend_slope, consistency, ...
  [KEPT] Cross-features v4    — mom3_x_tier, accel_x_viral, ...
  [KEPT] Strict chronological split — Train 11-26 / Val 27-28 / Test 29-30

  [ARCHITECTURE] Simplified stacking vs v5 :
      Base learners : XGBoost + LightGBM + CatBoost (RandomizedSearchCV)
      Meta-learner  : Ridge (L1 only — Meta-XGB removed because unstable on small meta-sets)
      → We keep the best of Ridge L1 and Stacking L2

Record to beat : R² = 0.3256 | MAE = 0.2959
Target         : R² > 0.34
"""

import ast
import os
import time
import warnings
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from PIL import Image, ImageFilter
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, KFold, cross_val_predict
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from textblob import TextBlob

warnings.filterwarnings("ignore")

RECORD_R2  = 0.3256
RECORD_MAE = 0.2959

print("=" * 70)
print("  Stacking v6 — Computer Vision Layer")
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
# [NEW 1] COMPUTER VISION LAYER — PIL + requests
# ─────────────────────────────────────────────────────────────────────
print("\n  Downloading & analyzing thumbnails (PIL + requests)...")

# ── CACHE : intermediate save every SAVE_EVERY images ──────
CV_CACHE_PATH = "cv_features_cache.csv"   # ← cache file on disk
SAVE_EVERY    = 100                        # ← save every 100 images
t_cv = time.time()

def extract_image_features(url, timeout=5):
    """
    Downloads the image from the URL and extracts 3 visual features.
    Returns (brightness, contrast, complexity) or None in case of error.

    - brightness  : average of grayscale pixels [0-255]
    - contrast    : pixel standard deviation (dispersion = contrast)
    - complexity  : variance of Laplacian filter (edges & textures)
    """
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None
        img = Image.open(io.BytesIO(resp.content)).convert("L")  # grayscale
        img_small = img.resize((64, 64))                          # size normalization
        arr = np.array(img_small, dtype=np.float32)

        brightness = float(arr.mean())
        contrast   = float(arr.std())

        # Complexity = Laplacian variance (edges & textures)
        lap        = img_small.filter(ImageFilter.FIND_EDGES)
        lap_arr    = np.array(lap, dtype=np.float32)
        complexity = float(lap_arr.var())

        return brightness, contrast, complexity
    except Exception:
        return None

# ── HOT RESTART : load existing cache if available ──────────
urls = df["coverUrl"].fillna("").tolist()
n    = len(urls)

if os.path.exists(CV_CACHE_PATH):
    cache_df   = pd.read_csv(CV_CACHE_PATH, index_col="row_idx")
    done_idx   = set(cache_df.index.tolist())
    print(f"    Cache found : {len(done_idx):,} images already processed out of {n:,}")
    print(f"    Resuming from index {max(done_idx)+1 if done_idx else 0}")
else:
    cache_df = pd.DataFrame(columns=["row_idx", "img_brightness", "img_contrast", "img_complexity"])
    cache_df = cache_df.set_index("row_idx")
    done_idx = set()
    print(f"    No cache — starting from scratch ({n:,} images to process)")

# ── DOWNLOAD LOOP with periodic saving ───────────────
pending = [i for i in range(n) if i not in done_idx]
errors  = 0
batch   = {}   # buffer before flushing to disk

print(f"   Remaining images : {len(pending):,} | Saving every {SAVE_EVERY} images\n")

for count, i in enumerate(pending):
    url  = urls[i]
    feat = None if not url else extract_image_features(url)

    if feat is None:
        errors += 1
        batch[i] = (np.nan, np.nan, np.nan)
    else:
        batch[i] = feat

    # Flush to CSV cache every SAVE_EVERY images
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
        eta = (elapsed / (count + 1)) * (len(pending) - count - 1)
        print(f"   [{total_done:>5}/{n}] new={count+1} | session_errors={errors}"
              f" | elapsed={elapsed:.0f}s | ETA≈{eta:.0f}s")

# ── RECONSTRUCTING THE FINAL DATAFRAME from the complete cache ────────
cache_df = pd.read_csv(CV_CACHE_PATH, index_col="row_idx").sort_index()

# Fallback : replace NaNs with the ensemble median
for col in ["img_brightness", "img_contrast", "img_complexity"]:
    median_val = cache_df[col].median()
    cache_df[col] = cache_df[col].fillna(median_val)

df["img_brightness"] = cache_df["img_brightness"].values
df["img_contrast"]   = cache_df["img_contrast"].values
df["img_complexity"] = cache_df["img_complexity"].values

total_errors = cache_df[["img_brightness"]].isna().sum().sum()  # after fillna = 0, but logged before
success = n - int(cache_df["img_brightness"].isna().sum())
print(f"\n   CV features ready in {time.time()-t_cv:.0f}s (current session)")
print(f"  Total cache   : {len(cache_df):,} lines → {CV_CACHE_PATH}")
print(f"  Brightness : mean={df['img_brightness'].mean():.1f} | std={df['img_brightness'].std():.1f}")
print(f"  Contrast   : mean={df['img_contrast'].mean():.1f} | std={df['img_contrast'].std():.1f}")
print(f"  Complexity : mean={df['img_complexity'].mean():.1f} | std={df['img_complexity'].std():.1f}")

CV_COLS = ["img_brightness", "img_contrast", "img_complexity"]

# ─────────────────────────────────────────────────────────────────────
# NLP LAYER (TextBlob) — identical to v5
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

print(f"   NLP features calculated in {time.time()-t_nlp:.1f}s")

NLP_COLS = ["sentiment_polarity", "sentiment_subjectivity", "emotional_intensity", "word_count"]

# ─────────────────────────────────────────────────────────────────────
# 3. MOMENTUM (identical to v4/v5)
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
# 4. CROSS-FEATURES (v4 + NLP + [NEW] CV crosses)
# ─────────────────────────────────────────────────────────────────────
print(" Calculating cross-features (Momentum × History + NLP + CV)...")

# v4 crosses
df["mom3_x_tier"]      = df["momentum_3"]    * df["follower_tier"]
df["accel_x_viral"]    = df["accel"]          * df["viral_potential"]
df["recovery_x_hist"]  = df["recovery"]       * df["hist_median_views"]
df["streak_x_engage"]  = df["streak_up"]      * df["engagement_total_hist"]
df["peak_x_p90"]       = df["peak_score"]     / (df["hist_p90_views"] + 1)
df["mom7_x_consist"]   = df["momentum_7"]     / (df["consistency"] + 0.1)
df["trend_x_duration"] = df["trend_slope"]    * df["duration"]
df["norm_x_tier"]      = df["momentum_norm"]  * df["follower_tier"]

# NLP crosses (v5)
df["intensity_x_momentum"] = df["emotional_intensity"] * df["momentum_3"]
df["polarity_x_viral"]     = df["sentiment_polarity"]  * df["viral_potential"]
df["subj_x_tier"]          = df["sentiment_subjectivity"] * df["follower_tier"]

# [NEW] CV crosses
df["contrast_x_momentum"]  = df["img_contrast"]    * df["momentum_3"]   # catchy thumbnail + strong momentum
df["bright_x_viral"]       = df["img_brightness"]  * df["viral_potential"]  # bright image + viral potential
df["complex_x_tier"]       = df["img_complexity"]  * df["follower_tier"]    # visual richness × audience size

CV_CROSS_COLS  = ["contrast_x_momentum", "bright_x_viral", "complex_x_tier"]
NLP_CROSS_COLS = ["intensity_x_momentum", "polarity_x_viral", "subj_x_tier"]

CROSS_COLS = [
    "mom3_x_tier", "accel_x_viral", "recovery_x_hist", "streak_x_engage",
    "peak_x_p90", "mom7_x_consist", "trend_x_duration", "norm_x_tier",
] + NLP_CROSS_COLS + CV_CROSS_COLS

df[CROSS_COLS] = df[CROSS_COLS].replace([np.inf, -np.inf], np.nan).fillna(df[CROSS_COLS].median())

# ─────────────────────────────────────────────────────────────────────
# 5. FINAL FEATURE SET
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
FEATURE_COLS = STATIC_FEATURES + MOMENTUM_COLS + CROSS_COLS + NLP_COLS + CV_COLS
TARGET = "target_log"

print(f"\n Feature set :")
print(f"   {len(STATIC_FEATURES)} static + {len(MOMENTUM_COLS)} momentum + {len(CROSS_COLS)} crossed")
print(f"   + {len(NLP_COLS)} NLP + {len(CV_COLS)} CV = {len(FEATURE_COLS)} total")
print(f"   [NEW] CV features  : {CV_COLS}")
print(f"   [NEW] CV crosses   : {CV_CROSS_COLS}")

# ─────────────────────────────────────────────────────────────────────
# 6. STRICT CHRONOLOGICAL SPLIT
# ─────────────────────────────────────────────────────────────────────
X_train = df.loc[df["video_rank"].between(11, 26), FEATURE_COLS].astype(float)
y_train = df.loc[df["video_rank"].between(11, 26), TARGET]
X_val   = df.loc[df["video_rank"].between(27, 28), FEATURE_COLS].astype(float)
y_val   = df.loc[df["video_rank"].between(27, 28), TARGET]
X_test  = df.loc[df["video_rank"].between(29, 30), FEATURE_COLS].astype(float)
y_test  = df.loc[df["video_rank"].between(29, 30), TARGET]

X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])
split_indices = np.array([-1] * len(X_train) + [0] * len(X_val))
ps = PredefinedSplit(split_indices)

print(f"\n Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")
print(f"   Record to beat : R²={RECORD_R2} | MAE={RECORD_MAE}\n")

# ─────────────────────────────────────────────────────────────────────
# 7. HELPER : tune + evaluate
# ─────────────────────────────────────────────────────────────────────
results = {}

def tune_and_eval(name, estimator, param_dist, n_iter=40, fit_kwargs=None):
    print(f"{'─'*70}")
    print(f"    {name}  (n_iter={n_iter})")
    print(f"{'─'*70}")
    t0 = time.time()

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=ps,
        scoring="neg_mean_absolute_error",
        refit=False,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_tv, y_tv, **(fit_kwargs or {}))
    best_params = search.best_params_
    print(f"  Best params : { {k: v for k, v in sorted(best_params.items())} }")

    final = estimator.__class__(**best_params, **{
        k: v for k, v in estimator.get_params().items()
        if k not in best_params
    })

    if isinstance(final, XGBRegressor):
        final.set_params(early_stopping_rounds=200, eval_metric="mae",
                         tree_method="hist", verbosity=0, random_state=42)
        final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        best_iter = final.best_iteration
    elif isinstance(final, LGBMRegressor):
        final.set_params(random_state=42, verbose=-1)
        callbacks = [lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)]
        final.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
        best_iter = final.best_iteration_
    elif isinstance(final, CatBoostRegressor):
        final.set_params(early_stopping_rounds=200, random_state=42, verbose=0)
        final.fit(X_train, y_train, eval_set=(X_val, y_val))
        best_iter = final.get_best_iteration()

    y_pred = final.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    elapsed = time.time() - t0

    print(f"  Best iteration   : {best_iter}")
    print(f"  R²  (test)       : {r2:.4f}  {'PASS' if r2 > RECORD_R2 else 'FAIL'}  (Δ {r2 - RECORD_R2:+.4f})")
    print(f"  MAE (test)       : {mae:.4f}  {'PASS' if mae < RECORD_MAE else 'FAIL'}  (Δ {mae - RECORD_MAE:+.4f})")
    print(f"  Time             : {elapsed:.0f}s\n")

    results[name] = {"model": final, "r2": r2, "mae": mae,
                     "params": best_params, "time": elapsed}
    return final

# ─────────────────────────────────────────────────────────────────────
# 8. XGBOOST
# ─────────────────────────────────────────────────────────────────────
xgb_params = {
    "max_depth":        [4, 5, 6, 7],
    "learning_rate":    [0.003, 0.005, 0.01],
    "n_estimators":     [5000, 8000, 10000],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8],
    "min_child_weight": [3, 5, 10],
    "reg_alpha":        [0.1, 0.5, 1.0],
    "reg_lambda":       [1.0, 2.0, 3.0],
    "gamma":            [0, 0.1, 0.3],
}
xgb_base  = XGBRegressor(tree_method="hist", early_stopping_rounds=200,
                          eval_metric="mae", verbosity=0, random_state=42)
xgb_model = tune_and_eval(
    "XGBoost", xgb_base, xgb_params, n_iter=50,
    fit_kwargs={"eval_set": [(X_val, y_val)]}
)

# ─────────────────────────────────────────────────────────────────────
# 9. LIGHTGBM
# ─────────────────────────────────────────────────────────────────────
lgbm_params = {
    "max_depth":         [-1, 5, 7],
    "learning_rate":     [0.005, 0.01, 0.02],
    "n_estimators":      [5000, 8000],
    "num_leaves":        [31, 63],
    "subsample":         [0.7, 0.8],
    "colsample_bytree":  [0.7, 0.8],
    "min_child_samples": [20, 50],
    "reg_alpha":         [0.1, 0.5],
    "reg_lambda":        [1.0, 2.0],
}
lgbm_base     = LGBMRegressor(random_state=42, verbose=-1)
callbacks_fit = [lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)]
lgbm_model    = tune_and_eval(
    "LightGBM", lgbm_base, lgbm_params, n_iter=25,
    fit_kwargs={"eval_set": [(X_val, y_val)], "callbacks": callbacks_fit}
)

# ─────────────────────────────────────────────────────────────────────
# 10. CATBOOST
# ─────────────────────────────────────────────────────────────────────
cat_params = {
    "depth":               [6, 8, 10],
    "learning_rate":       [0.003, 0.005, 0.01],
    "iterations":          [5000, 8000, 10000],
    "l2_leaf_reg":         [3, 5, 10, 15],
    "subsample":           [0.7, 0.8, 0.9],
    "colsample_bylevel":   [0.6, 0.7, 0.8],
    "min_data_in_leaf":    [5, 10, 20],
    "bagging_temperature": [0.0, 0.3, 0.7],
    "random_strength":     [1, 3, 5],
}
cat_base  = CatBoostRegressor(early_stopping_rounds=200, random_state=42, verbose=0)
cat_model = tune_and_eval(
    "CatBoost", cat_base, cat_params, n_iter=80,
    fit_kwargs={"eval_set": (X_val, y_val)}
)

# ─────────────────────────────────────────────────────────────────────
# 11. STACKING L1 — Ridge meta-learner
# ─────────────────────────────────────────────────────────────────────
print("─" * 70)
print("    Stacking L1 — Ridge meta-learner")
print("─" * 70)
t0 = time.time()

xgb_s  = XGBRegressor(**results["XGBoost"]["params"],
                       tree_method="hist", early_stopping_rounds=200,
                       eval_metric="mae", verbosity=0, random_state=42)
lgbm_s = LGBMRegressor(**results["LightGBM"]["params"], random_state=42, verbose=-1)
cat_s  = CatBoostRegressor(**results["CatBoost"]["params"],
                            early_stopping_rounds=200, random_state=42, verbose=0)

xgb_s.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
lgbm_s.fit(X_train, y_train, eval_set=[(X_val, y_val)],
           callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)])
cat_s.fit(X_train, y_train, eval_set=(X_val, y_val))

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
print(f"  R²  (test) : {r2_ridge:.4f}  {'PASS' if r2_ridge > RECORD_R2 else 'FAIL'}  (Δ {r2_ridge - RECORD_R2:+.4f})")
print(f"  MAE (test) : {mae_ridge:.4f}  {'PASS' if mae_ridge < RECORD_MAE else 'FAIL'}  (Δ {mae_ridge - RECORD_MAE:+.4f})")
print(f"  Time       : {elapsed:.0f}s\n")
results["Stacking-Ridge"] = {"r2": r2_ridge, "mae": mae_ridge, "time": elapsed}

# ─────────────────────────────────────────────────────────────────────
# 12. STACKING L2 — OOF 5-fold
# ─────────────────────────────────────────────────────────────────────
print("─" * 70)
print("    Stacking L2 — OOF 5-fold")
print("─" * 70)
t0 = time.time()

kf = KFold(n_splits=5, shuffle=False)

oof_xgb  = cross_val_predict(
    XGBRegressor(**results["XGBoost"]["params"],
                 tree_method="hist", verbosity=0, random_state=42),
    X_train, y_train, cv=kf
)
oof_lgbm = cross_val_predict(
    LGBMRegressor(**results["LightGBM"]["params"], random_state=42, verbose=-1),
    X_train, y_train, cv=kf
)
oof_cat  = cross_val_predict(
    CatBoostRegressor(**results["CatBoost"]["params"], random_state=42, verbose=0),
    X_train, y_train, cv=kf
)

meta_train_l2 = build_meta_features(oof_xgb, oof_lgbm, oof_cat)
meta_val_l2   = meta_val_feats
meta_test_l2  = meta_test_feats

xgb_l2 = XGBRegressor(
    max_depth=4, learning_rate=0.01, n_estimators=3000,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5,
    early_stopping_rounds=100, eval_metric="mae",
    tree_method="hist", verbosity=0, random_state=77
)
xgb_l2.fit(meta_train_l2, y_train, eval_set=[(meta_val_l2, y_val)], verbose=False)

cat_l2 = CatBoostRegressor(
    depth=5, learning_rate=0.01, iterations=3000,
    l2_leaf_reg=5, subsample=0.8,
    early_stopping_rounds=100, random_state=77, verbose=0
)
cat_l2.fit(meta_train_l2, y_train, eval_set=(meta_val_l2, y_val))

meta_l2_val  = np.column_stack([xgb_l2.predict(meta_val_l2),  cat_l2.predict(meta_val_l2)])
meta_l2_test = np.column_stack([xgb_l2.predict(meta_test_l2), cat_l2.predict(meta_test_l2)])

ridge_l2 = Ridge(alpha=1.0)
ridge_l2.fit(meta_l2_val, y_val)
y_pred_l2  = ridge_l2.predict(meta_l2_test)
r2_l2      = r2_score(y_test, y_pred_l2)
mae_l2     = mean_absolute_error(y_test, y_pred_l2)
elapsed    = time.time() - t0

print(f"  R²  (test) : {r2_l2:.4f}  {'PASS' if r2_l2 > RECORD_R2 else 'FAIL'}  (Δ {r2_l2 - RECORD_R2:+.4f})")
print(f"  MAE (test) : {mae_l2:.4f}  {'PASS' if mae_l2 < RECORD_MAE else 'FAIL'}  (Δ {mae_l2 - RECORD_MAE:+.4f})")
print(f"  Time       : {elapsed:.0f}s\n")
results["Stacking-L2"] = {"r2": r2_l2, "mae": mae_l2, "time": elapsed}

# ─────────────────────────────────────────────────────────────────────
# 13. FEATURE IMPORTANCE — CV focus vs other layers
# ─────────────────────────────────────────────────────────────────────
print("─" * 70)
print("   Feature Importance (CatBoost) — top 25 + CV analysis")
print("─" * 70)

fi = pd.Series(cat_s.get_feature_importance(), index=FEATURE_COLS).sort_values(ascending=False)

print(f"\n  {'Feature':<35} {'Imp':>6}  {'Type'}")
print(f"  {'─'*62}")
for feat, imp in fi.head(25).items():
    if feat in CV_COLS:          tag = " ←   CV"
    elif feat in CV_CROSS_COLS:  tag = " ←   CV×"
    elif feat in NLP_COLS:       tag = " ←  NLP"
    elif feat in NLP_CROSS_COLS: tag = " ←  NLP×"
    elif feat in CROSS_COLS:     tag = " ← CROSSED"
    elif feat in MOMENTUM_COLS:  tag = " ← momentum"
    else:                        tag = ""
    print(f"  {feat:<35} {imp:>6.2f}  {tag}")

# Importance by layer
cv_total       = fi[CV_COLS].sum()
cv_cross_total = fi[CV_CROSS_COLS].sum()
nlp_total      = fi[NLP_COLS].sum()
nlp_cross_total = fi[NLP_CROSS_COLS].sum()
cross_base_total = fi[[c for c in CROSS_COLS if c not in NLP_CROSS_COLS and c not in CV_CROSS_COLS]].sum()
momentum_total = fi[MOMENTUM_COLS].sum()
static_total   = fi[STATIC_FEATURES].sum()

print(f"\n  ── Importance by layer ────────────────────────────")
print(f"  Static              : {static_total:.2f}%")
print(f"  Momentum            : {momentum_total:.2f}%")
print(f"  Crossed v4          : {cross_base_total:.2f}%")
print(f"  NLP (TextBlob)      : {nlp_total:.2f}%")
print(f"  NLP × Cross         : {nlp_cross_total:.2f}%")
print(f"  CV  (PIL)           : {cv_total:.2f}%  ← new")
print(f"  CV  × Cross         : {cv_cross_total:.2f}%  ← new")

# Key question : img_contrast vs followers ?
ic_rank = (fi.index.tolist().index("img_contrast") + 1) if "img_contrast" in fi.index else "N/A"
fo_rank = (fi.index.tolist().index("followers")    + 1) if "followers"    in fi.index else "N/A"
ic_imp  = fi.get("img_contrast", 0)
fo_imp  = fi.get("followers",    0)

print(f"\n  ── Answer to key question ─────────────────────────")
print(f"  img_contrast : rank #{ic_rank:<4}  importance = {ic_imp:.2f}%")
print(f"  followers    : rank #{fo_rank:<4}  importance = {fo_imp:.2f}%")
if isinstance(ic_rank, int) and isinstance(fo_rank, int) and ic_rank < fo_rank:
    print(f"   img_contrast EXCEEDS followers (#{ic_rank} vs #{fo_rank})")
else:
    print(f"   img_contrast below followers (#{ic_rank} vs #{fo_rank}) — CV present but not dominant")

# ─────────────────────────────────────────────────────────────────────
# 14. SUMMARY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL SUMMARY — v6 CV")
print("=" * 70)
print(f"\n{'Model':<22} {'R²':>8} {'Δ R²':>9} {'MAE':>8} {'Δ MAE':>9}")
print("─" * 62)
for name, res in results.items():
    dr2  = res["r2"]  - RECORD_R2
    dmae = res["mae"] - RECORD_MAE
    icon = "* " if res["r2"] == max(r["r2"] for r in results.values()) else "  "
    print(f"{icon} {name:<20} {res['r2']:>8.4f} {dr2:>+9.4f} {res['mae']:>8.4f} {dmae:>+9.4f}")
print("─" * 62)
print(f"   {'Record v3':20} {RECORD_R2:>8.4f} {'':>9} {RECORD_MAE:>8.4f}")

best_name = max(results, key=lambda k: results[k]["r2"])
best_r2   = results[best_name]["r2"]
print(f"\n Best : {best_name}  (R²={best_r2:.4f})")
if best_r2 > 0.34:
    print(" TARGET R² > 0.34 ACHIEVED !")
elif best_r2 > RECORD_R2:
    print(f" New record ! Gain : +{best_r2 - RECORD_R2:.4f}")
else:
    print(" Persistent plateau — next step: Target Encoding by creator")

# ─────────────────────────────────────────────────────────────────────
# 15. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle("Stacking v6 — Computer Vision Layer (PIL + requests)", fontsize=15,
             color="white", fontweight="bold", y=1.01)

P = dict(ax_bg="#1a1a1a", grid="#2a2a2a", text="#f0f0f0",
         gold="#ffbe0b", record="#ff4d6d",
         mom="#00f5d4", cross="#ff9f1c", stat="#666666",
         nlp="#c77dff", nlp_cross="#e0aaff",
         cv="#ff6b6b", cv_cross="#ffa07a")

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
    bars[best_idx].set_edgecolor(P["gold"]); bars[best_idx].set_linewidth(2.5); bars[best_idx].set_alpha(1.0)
    record_val = RECORD_R2 if metric == "R²" else RECORD_MAE
    ax.axhline(record_val, color=P["record"], linewidth=1.5, linestyle="--", zorder=4,
               label=f"Record v3 ({record_val:.4f})")
    if metric == "R²":
        ax.axhline(0.34, color="#00b4d8", linewidth=1.0, linestyle=":", zorder=4, label="Target 0.34")
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

# Feature importance : top 20, colored by layer
ax = axes[2]
ax.set_facecolor(P["ax_bg"])
top20 = fi.head(20)

def feat_color(f):
    if f in CV_COLS:          return P["cv"]
    if f in CV_CROSS_COLS:    return P["cv_cross"]
    if f in NLP_COLS:         return P["nlp"]
    if f in NLP_CROSS_COLS:   return P["nlp_cross"]
    if f in CROSS_COLS:       return P["cross"]
    if f in MOMENTUM_COLS:    return P["mom"]
    return P["stat"]

fi_colors = [feat_color(f) for f in top20.index]
ax.barh(range(len(top20)), top20.values, color=fi_colors, alpha=0.88, zorder=3)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.index, fontsize=7.5, color=P["text"])
ax.invert_yaxis()
ax.set_title("Feature Importance (CatBoost)", color=P["text"], fontsize=11, fontweight="bold", pad=10)
ax.tick_params(colors=P["text"], labelsize=8)
ax.xaxis.grid(True, color=P["grid"], linewidth=0.8, zorder=0)
for sp in ax.spines.values(): sp.set_edgecolor(P["grid"])
legend_handles = [
    mpatches.Patch(color=P["cv"],        label=f"CV PIL        ({cv_total:.1f}%)"),
    mpatches.Patch(color=P["cv_cross"],  label=f"CV × Cross    ({cv_cross_total:.1f}%)"),
    mpatches.Patch(color=P["nlp"],       label=f"NLP TextBlob  ({nlp_total:.1f}%)"),
    mpatches.Patch(color=P["nlp_cross"], label=f"NLP × Cross   ({nlp_cross_total:.1f}%)"),
    mpatches.Patch(color=P["cross"],     label=f"Crossed v4    ({cross_base_total:.1f}%)"),
    mpatches.Patch(color=P["mom"],       label=f"Momentum      ({momentum_total:.1f}%)"),
    mpatches.Patch(color=P["stat"],      label=f"Static        ({static_total:.1f}%)"),
]
ax.legend(handles=legend_handles, framealpha=0.2, labelcolor=P["text"], fontsize=7, loc="lower right")

plt.tight_layout()
out_path = "stacking_v6_cv.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n Chart saved → {out_path}")
plt.show()
print("\n Stacking v6 CV completed.\n")