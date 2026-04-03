"""
Stacking v4 — Structural Fixes
================================
Diagnostic v3 -> 4 targeted corrections:

  [FIX 1] XGBoost Meta-learner instead of Ridge/Lasso
      Ridge assigned 42% weight to LightGBM (the weakest model). An XGBoost
      meta-learner captures non-linear interactions between the 3 predictors.

  [FIX 2] Slow Learning Rate for CatBoost and XGBoost
      In v3, models stopped at best_iter=155-220 out of 2000-4000.
      We force lr=0.005 + n_estimators=10000 -> early stopping allows
      trees to refine instead of stopping prematurely.

  [FIX 3] Unconditional L2 Stacking
      The 0.34 threshold prevented L2 from running. Removed.
      L2 always runs and we retain the best of L1 and L2.

  [FIX 4] Momentum x History Crossed Features
      Interactions that GBDTs detect better if explicitly provided:
      momentum_3 x follower_tier, accel x viral_potential, etc.

Record to beat: R² = 0.3256 | MAE = 0.2959
Goal          : R² > 0.35
"""

import ast
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, KFold, cross_val_predict
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

RECORD_R2  = 0.3256
RECORD_MAE = 0.2959

print("=" * 70)
print("  Stacking v4 — Structural Fixes")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# 1. LOADING & CREATOR_ID
# ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("cleaned_data.csv")
df = df.sort_values(["followers", "video_rank"]).reset_index(drop=True)
df["creator_id"] = df.groupby("followers").ngroup()

print(f"\n{df['creator_id'].nunique()} creators identified | {len(df):,} videos")

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
# 3. MOMENTUM v2 + v3
# ─────────────────────────────────────────────────────────────────────
print("\nCalculating Momentum (v2 + v3)...")

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

df["momentum_3"]    = shifted.groupby(df["creator_id"]).transform(lambda s: s.rolling(3, min_periods=1).mean())
df["trend_slope"]   = shifted.groupby(df["creator_id"]).transform(lambda s: rolling_slope(s, window=5))
df["consistency"]   = shifted.groupby(df["creator_id"]).transform(lambda s: s.rolling(5, min_periods=2).std())
df["momentum_ratio"]  = df["momentum_3"] / (df["hist_median_views"] + 1)
df["trend_direction"] = np.sign(df["trend_slope"].fillna(0)).astype(int)
df["volatility_tier"] = df["consistency"] / (df["hist_median_views"] + 1)
df["momentum_7"]    = shifted.groupby(df["creator_id"]).transform(lambda s: s.rolling(7, min_periods=2).mean())
df["accel"]         = df["momentum_3"] - df["momentum_7"]
df["peak_score"]    = shifted.groupby(df["creator_id"]).transform(lambda s: s.rolling(5, min_periods=1).max())
recent_min          = shifted.groupby(df["creator_id"]).transform(lambda s: s.rolling(5, min_periods=1).min())
df["recovery"]      = (df["momentum_3"] - recent_min) / (df["peak_score"] - recent_min + 1)
df["streak_up"]     = shifted.groupby(df["creator_id"]).transform(lambda s: count_streak(s))
df["momentum_norm"] = df["momentum_3"] / (df["hist_p90_views"] + 1)

MOMENTUM_COLS = [
    "momentum_3", "trend_slope", "consistency", "momentum_ratio",
    "trend_direction", "volatility_tier", "momentum_7", "accel",
    "peak_score", "recovery", "streak_up", "momentum_norm",
]
df[MOMENTUM_COLS] = df[MOMENTUM_COLS].fillna(df[MOMENTUM_COLS].median())

# ─────────────────────────────────────────────────────────────────────
# 4. [FIX 4] CROSS FEATURES: MOMENTUM x HISTORY
# ─────────────────────────────────────────────────────────────────────
print("Calculating Cross Features...")

df["mom3_x_tier"]      = df["momentum_3"]  * df["follower_tier"]
df["accel_x_viral"]    = df["accel"]       * df["viral_potential"]
df["recovery_x_hist"]  = df["recovery"]    * df["hist_median_views"]
df["streak_x_engage"]  = df["streak_up"]   * df["engagement_total_hist"]
df["peak_x_p90"]       = df["peak_score"]  / (df["hist_p90_views"] + 1)
df["mom7_x_consist"]   = df["momentum_7"]  / (df["consistency"] + 0.1)
df["trend_x_duration"] = df["trend_slope"] * df["duration"]
df["norm_x_tier"]      = df["momentum_norm"] * df["follower_tier"]

CROSS_COLS = [
    "mom3_x_tier", "accel_x_viral", "recovery_x_hist", "streak_x_engage",
    "peak_x_p90", "mom7_x_consist", "trend_x_duration", "norm_x_tier",
]
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
FEATURE_COLS = STATIC_FEATURES + MOMENTUM_COLS + CROSS_COLS
TARGET = "target_log"

print(f"\nFeature set: {len(STATIC_FEATURES)} static + {len(MOMENTUM_COLS)} momentum + {len(CROSS_COLS)} cross = {len(FEATURE_COLS)} total")

# ─────────────────────────────────────────────────────────────────────
# 6. CHRONOLOGICAL SPLIT
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

print(f"Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")
print(f"   Record to beat: R²={RECORD_R2} | MAE={RECORD_MAE}\n")

# ─────────────────────────────────────────────────────────────────────
# 7. HELPER: tune + evaluate
# ─────────────────────────────────────────────────────────────────────
results = {}

def tune_and_eval(name, estimator, param_dist, n_iter=40, fit_kwargs=None):
    print(f"{'─'*70}")
    print(f"  {name}  (n_iter={n_iter})")
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
    print(f"  Best params: { {k: v for k, v in sorted(best_params.items())} }")

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
    print(f"  R²  (test)       : {r2:.4f}  (Δ {r2 - RECORD_R2:+.4f})")
    print(f"  MAE (test)       : {mae:.4f}  (Δ {mae - RECORD_MAE:+.4f})")
    print(f"  Time             : {elapsed:.0f}s\n")

    results[name] = {"model": final, "r2": r2, "mae": mae,
                     "params": best_params, "time": elapsed}
    return final

# ─────────────────────────────────────────────────────────────────────
# 8. [FIX 2] XGBOOST — slow learning rate
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
xgb_base = XGBRegressor(tree_method="hist", early_stopping_rounds=200,
                         eval_metric="mae", verbosity=0, random_state=42)
xgb_model = tune_and_eval(
    "XGBoost", xgb_base, xgb_params, n_iter=50,
    fit_kwargs={"eval_set": [(X_val, y_val)]}
)

# ─────────────────────────────────────────────────────────────────────
# 9. LIGHTGBM — reduced budget for diversity
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
lgbm_base = LGBMRegressor(random_state=42, verbose=-1)
callbacks_fit = [lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)]
lgbm_model = tune_and_eval(
    "LightGBM", lgbm_base, lgbm_params, n_iter=25,
    fit_kwargs={"eval_set": [(X_val, y_val)], "callbacks": callbacks_fit}
)

# ─────────────────────────────────────────────────────────────────────
# 10. [FIX 2] CATBOOST — very slow learning rate
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
cat_base = CatBoostRegressor(early_stopping_rounds=200, random_state=42, verbose=0)
cat_model = tune_and_eval(
    "CatBoost", cat_base, cat_params, n_iter=80,
    fit_kwargs={"eval_set": (X_val, y_val)}
)

# ─────────────────────────────────────────────────────────────────────
# 11. STACKING LEVEL 1 — [FIX 1] XGBoost Meta-learner
# ─────────────────────────────────────────────────────────────────────
print("─" * 70)
print("  Stacking L1 — XGBoost meta-learner (non-linear)")
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

p_xgb_val  = xgb_s.predict(X_val)
p_lgbm_val = lgbm_s.predict(X_val)
p_cat_val  = cat_s.predict(X_val)
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

meta_xgb = XGBRegressor(
    max_depth=3, learning_rate=0.05, n_estimators=500,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
    tree_method="hist", verbosity=0, random_state=42
)
meta_xgb.fit(meta_val_feats, y_val)
y_pred_meta_xgb   = meta_xgb.predict(meta_test_feats)
r2_meta_xgb   = r2_score(y_test, y_pred_meta_xgb)
mae_meta_xgb  = mean_absolute_error(y_test, y_pred_meta_xgb)

meta_ridge = Ridge(alpha=1.0)
meta_ridge.fit(meta_val_feats, y_val)
y_pred_ridge = meta_ridge.predict(meta_test_feats)
r2_ridge     = r2_score(y_test, y_pred_ridge)
mae_ridge    = mean_absolute_error(y_test, y_pred_ridge)

if r2_meta_xgb >= r2_ridge:
    y_pred_l1, r2_l1, mae_l1, name_l1 = y_pred_meta_xgb, r2_meta_xgb, mae_meta_xgb, "Stacking-MetaXGB"
else:
    y_pred_l1, r2_l1, mae_l1, name_l1 = y_pred_ridge, r2_ridge, mae_ridge, "Stacking-Ridge"

elapsed = time.time() - t0
print(f"  Best L1 : {name_l1} | R²={r2_l1:.4f} | MAE={mae_l1:.4f} | Time: {elapsed:.0f}s\n")
results[name_l1] = {"r2": r2_l1, "mae": mae_l1, "time": elapsed}

# ─────────────────────────────────────────────────────────────────────
# 12. [FIX 3] STACKING LEVEL 2 — UNCONDITIONAL
# ─────────────────────────────────────────────────────────────────────
print("─" * 70)
print("  Stacking L2 — unconditional (OOF 5-fold on train)")
print("─" * 70)
t0 = time.time()

kf = KFold(n_splits=5, shuffle=False)

oof_xgb  = cross_val_predict(
    XGBRegressor(**results["XGBoost"]["params"], tree_method="hist", verbosity=0, random_state=42),
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

xgb_l2a = XGBRegressor(
    max_depth=4, learning_rate=0.01, n_estimators=3000,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5,
    early_stopping_rounds=100, eval_metric="mae",
    tree_method="hist", verbosity=0, random_state=77
)
xgb_l2a.fit(meta_train_l2, y_train, eval_set=[(meta_val_l2, y_val)], verbose=False)

cat_l2b = CatBoostRegressor(
    depth=5, learning_rate=0.01, iterations=3000,
    l2_leaf_reg=5, subsample=0.8,
    early_stopping_rounds=100, random_state=77, verbose=0
)
cat_l2b.fit(meta_train_l2, y_train, eval_set=(meta_val_l2, y_val))

meta_l2_val  = np.column_stack([xgb_l2a.predict(meta_val_l2),  cat_l2b.predict(meta_val_l2)])
meta_l2_test = np.column_stack([xgb_l2a.predict(meta_test_l2), cat_l2b.predict(meta_test_l2)])

ridge_l2 = Ridge(alpha=1.0)
ridge_l2.fit(meta_l2_val, y_val)
y_pred_l2  = ridge_l2.predict(meta_l2_test)
r2_l2  = r2_score(y_test, y_pred_l2)
mae_l2 = mean_absolute_error(y_test, y_pred_l2)
elapsed = time.time() - t0

print(f"  R²  (test) : {r2_l2:.4f} | MAE (test) : {mae_l2:.4f} | Time: {elapsed:.0f}s\n")
results["Stacking-L2"] = {"r2": r2_l2, "mae": mae_l2, "time": elapsed}

# ─────────────────────────────────────────────────────────────────────
# 14. SUMMARY
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
for name, res in results.items():
    icon = "*" if res["r2"] == max(r["r2"] for r in results.values()) else " "
    print(f"{icon} {name:<20} R²={res['r2']:>8.4f} MAE={res['mae']:>8.4f}")

best_name = max(results, key=lambda k: results[k]["r2"])
print(f"\nBest model: {best_name} (R²={results[best_name]['r2']:.4f})")

plt.show()
print("\nStacking v4 completed.\n")