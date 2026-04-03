"""
Stacking 3 — Push Beyond 0.35
================================
Strategies compared to v3 (R²=0.3256):

  [A] EXTENDED MOMENTUM
      • momentum_7  : 7-video moving average (slow trends)
      • accel       : 2nd derivative of momentum (acceleration/deceleration)
      • peak_score  : best score in the last 5 videos
      • recovery    : rebound after a low (current score vs recent min)

  [B] ASYMMETRIC TUNING BUDGET
      • CatBoost : n_iter=80 (dominates at 59%, deserves the budget)
      • XGBoost  : n_iter=50
      • LightGBM : n_iter=30

  [C] LASSO META-MODEL
      • Lasso(cv) forces a sharper model selection
      • Ridge compared as fallback

  [D] TWO-LEVEL STACKING (optional, activated if level1 > 0.34)
      • Level 1 : XGB + LGBM + CAT  -> OOF predictions on val
      • Level 2 : XGB_light + CAT_light trained on these OOF
      • Meta     : Lasso on level 2 outputs

Record to beat: R² = 0.3256 | MAE = 0.2959
"""

import ast
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.linear_model import Ridge, LassoCV
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

RECORD_R2  = 0.3256
RECORD_MAE = 0.2959

print("=" * 70)
print("  Stacking 3 — Push Beyond 0.35")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# 1. LOADING & CREATOR_ID
# ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("cleaned_data.csv")
df = df.sort_values(["followers", "video_rank"]).reset_index(drop=True)
df["creator_id"] = df.groupby("followers").ngroup()

n_creators = df["creator_id"].nunique()
print(f"\n{n_creators} creators | {len(df):,} videos")

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
# 3. DYNAMIC FEATURE ENGINEERING v3 — EXTENDED MOMENTUM
#    Absolute rule: .shift(1) before any rolling -> zero leakage
# ─────────────────────────────────────────────────────────────────────
print("\nCalculating Extended Momentum (v3)...")

def rolling_slope(series, window=5):
    """Linear slope (polyfit) over a rolling window."""
    def slope(arr):
        if arr.isna().any() or len(arr) < 2:
            return np.nan
        x = np.arange(len(arr))
        return np.polyfit(x, arr.values, 1)[0]
    return series.rolling(window, min_periods=2).apply(slope, raw=False)

grp = df.groupby("creator_id")["explosion_score"]
shifted = grp.shift(1)

# -- v2 features (preserved) --------------------------------------
df["momentum_3"] = (
    shifted.groupby(df["creator_id"])
    .transform(lambda s: s.rolling(3, min_periods=1).mean())
)
df["trend_slope"] = (
    shifted.groupby(df["creator_id"])
    .transform(lambda s: rolling_slope(s, window=5))
)
df["consistency"] = (
    shifted.groupby(df["creator_id"])
    .transform(lambda s: s.rolling(5, min_periods=2).std())
)
df["momentum_ratio"]    = df["momentum_3"] / (df["hist_median_views"] + 1)
df["trend_direction"]   = np.sign(df["trend_slope"].fillna(0)).astype(int)
df["volatility_tier"]   = df["consistency"] / (df["hist_median_views"] + 1)

# -- v3 features (new) ---------------------------------------
# momentum_7 : slow trend over 7 videos
df["momentum_7"] = (
    shifted.groupby(df["creator_id"])
    .transform(lambda s: s.rolling(7, min_periods=2).mean())
)

# accel : difference between momentum_3 and momentum_7
df["accel"] = df["momentum_3"] - df["momentum_7"]

# peak_score : best score over the last 5 videos
df["peak_score"] = (
    shifted.groupby(df["creator_id"])
    .transform(lambda s: s.rolling(5, min_periods=1).max())
)

# recovery : current score vs recent low (potential rebound)
recent_min = (
    shifted.groupby(df["creator_id"])
    .transform(lambda s: s.rolling(5, min_periods=1).min())
)
df["recovery"] = (df["momentum_3"] - recent_min) / (df["peak_score"] - recent_min + 1)

# streak_up : number of consecutive videos rising
def count_streak(series):
    def streak(arr):
        n = 0
        for i in range(len(arr) - 1, 0, -1):
            if arr[i] > arr[i-1]:
                n += 1
            else:
                break
        return n
    return series.rolling(5, min_periods=1).apply(streak, raw=True)

df["streak_up"] = (
    shifted.groupby(df["creator_id"])
    .transform(lambda s: count_streak(s))
)

# momentum_norm : momentum_3 normalized by p90
df["momentum_norm"] = df["momentum_3"] / (df["hist_p90_views"] + 1)

MOMENTUM_V2 = [
    "momentum_3", "trend_slope", "consistency",
    "momentum_ratio", "trend_direction", "volatility_tier",
]
MOMENTUM_V3 = [
    "momentum_7", "accel", "peak_score",
    "recovery", "streak_up", "momentum_norm",
]
ALL_MOMENTUM = MOMENTUM_V2 + MOMENTUM_V3

# Median imputation
df[ALL_MOMENTUM] = df[ALL_MOMENTUM].fillna(df[ALL_MOMENTUM].median())

print(f"  {len(MOMENTUM_V2)} v2 features + {len(MOMENTUM_V3)} v3 features = {len(ALL_MOMENTUM)} total momentum")

# ─────────────────────────────────────────────────────────────────────
# 4. FINAL FEATURE SET
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
FEATURE_COLS = STATIC_FEATURES + ALL_MOMENTUM
TARGET = "target_log"

# ─────────────────────────────────────────────────────────────────────
# 5. SPLIT CHRONOLOGIQUE
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

# ─────────────────────────────────────────────────────────────────────
# 6. HELPER : tune + evaluate
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

    final = estimator.__class__(**best_params, **{
        k: v for k, v in estimator.get_params().items()
        if k not in best_params
    })

    if isinstance(final, XGBRegressor):
        final.set_params(early_stopping_rounds=100, eval_metric="mae",
                         tree_method="hist", verbosity=0, random_state=42)
        final.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        best_iter = final.best_iteration
    elif isinstance(final, LGBMRegressor):
        final.set_params(n_estimators=best_params.get("n_estimators", 2000),
                         random_state=42, verbose=-1)
        callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
        final.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
        best_iter = final.best_iteration_
    elif isinstance(final, CatBoostRegressor):
        final.set_params(early_stopping_rounds=100, random_state=42, verbose=0)
        final.fit(X_train, y_train, eval_set=(X_val, y_val))
        best_iter = final.get_best_iteration()

    y_pred = final.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    elapsed = time.time() - t0

    print(f"  R² (test): {r2:.4f} | MAE (test): {mae:.4f} | Time: {elapsed:.0f}s\n")

    results[name] = {"model": final, "r2": r2, "mae": mae,
                     "params": best_params, "time": elapsed}
    return final

# ─────────────────────────────────────────────────────────────────────
# 7, 8, 9. MODELS TUNING
# ─────────────────────────────────────────────────────────────────────
xgb_params = {"max_depth": [5, 7, 9], "learning_rate": [0.01, 0.03, 0.05], "n_estimators": [2000, 3000, 4000]}
xgb_model = tune_and_eval("XGBoost", XGBRegressor(tree_method="hist", random_state=42), xgb_params, n_iter=50, fit_kwargs={"eval_set": [(X_val, y_val)]})

lgbm_params = {"max_depth": [5, 7, 9, -1], "learning_rate": [0.005, 0.01, 0.03, 0.05], "n_estimators": [2000, 3000, 5000]}
lgbm_model = tune_and_eval("LightGBM", LGBMRegressor(random_state=42), lgbm_params, n_iter=30, fit_kwargs={"eval_set": [(X_val, y_val)], "callbacks": [lgb.early_stopping(100, verbose=False)]})

cat_params = {"depth": [4, 6, 8, 10], "learning_rate": [0.01, 0.03, 0.05], "iterations": [1000, 2000, 3000]}
cat_model = tune_and_eval("CatBoost", CatBoostRegressor(random_state=42, verbose=0), cat_params, n_iter=80, fit_kwargs={"eval_set": (X_val, y_val)})

# ─────────────────────────────────────────────────────────────────────
# 10. STACKING LEVEL 1
# ─────────────────────────────────────────────────────────────────────
print("─" * 70)
print("  Stacking L1 — Lasso vs Ridge meta-learner")
print("─" * 70)

cat_s = cat_model # Base for feature importance

meta_val = np.column_stack([xgb_model.predict(X_val), lgbm_model.predict(X_val), cat_model.predict(X_val)])
meta_test = np.column_stack([xgb_model.predict(X_test), lgbm_model.predict(X_test), cat_model.predict(X_test)])

lasso_meta = LassoCV(cv=5, random_state=42).fit(meta_val, y_val)
y_pred_l1 = lasso_meta.predict(meta_test)
r2_l1 = r2_score(y_test, y_pred_l1)
mae_l1 = mean_absolute_error(y_test, y_pred_l1)

results["Stacking-Lasso"] = {"r2": r2_l1, "mae": mae_l1, "time": 5}

# ─────────────────────────────────────────────────────────────────────
# 12. FEATURE IMPORTANCE MOMENTUM — v2 vs v3
# ─────────────────────────────────────────────────────────────────────
print("─" * 70)
print("  Momentum Feature Importance v2 vs v3 (CatBoost)")
print("─" * 70)

fi = pd.Series(cat_s.get_feature_importance(), index=FEATURE_COLS).sort_values(ascending=False)
fi_momentum_v2 = fi[MOMENTUM_V2].sort_values(ascending=False)
fi_momentum_v3 = fi[MOMENTUM_V3].sort_values(ascending=False)

print(f"\n  Total importance v2 : {fi_momentum_v2.sum():.2f}")
print(f"  Total importance v3 : {fi_momentum_v3.sum():.2f}")

# ─────────────────────────────────────────────────────────────────────
# 14. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 10))
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle("Stacking v3 — Push Beyond 0.35", fontsize=15, color="white", fontweight="bold", y=1.01)

P = dict(ax_bg="#1a1a1a", grid="#2a2a2a", text="#f0f0f0", gold="#ffbe0b", record="#ff4d6d", v2="#00f5d4", v3="#ff9f1c")

gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)
ax_r2  = fig.add_subplot(gs[0, 0])
ax_fi  = fig.add_subplot(gs[0, 2])
ax_mom = fig.add_subplot(gs[1, :2])

model_names = list(results.keys())
r2_vals     = [results[n]["r2"]  for n in model_names]

# — R²
ax_r2.set_facecolor(P["ax_bg"])
ax_r2.bar(model_names, r2_vals, color="#3a86ff", alpha=0.88)
ax_r2.set_title("R²", color=P["text"], fontsize=13, fontweight="bold")
ax_r2.axhline(RECORD_R2, color=P["record"], linestyle="--", label=f"Record v3 ({RECORD_R2:.4f})")
ax_r2.legend()

# — Feature Importance
ax_fi.set_facecolor(P["ax_bg"])
top15 = fi.head(15)
fi_colors = [P["v3"] if f in MOMENTUM_V3 else P["v2"] if f in MOMENTUM_V2 else "#666666" for f in top15.index]
ax_fi.barh(range(len(top15)), top15.values, color=fi_colors)
ax_fi.set_yticks(range(len(top15)))
ax_fi.set_yticklabels(top15.index, color=P["text"])
ax_fi.invert_yaxis()
ax_fi.set_title("Feature Importance (CatBoost)", color=P["text"])

# — Momentum Comparison
ax_mom.set_facecolor(P["ax_bg"])
all_mom_feats = MOMENTUM_V2 + MOMENTUM_V3
mom_imps      = [fi.get(f, 0) for f in all_mom_feats]
mom_colors    = [P["v2"]] * len(MOMENTUM_V2) + [P["v3"]] * len(MOMENTUM_V3)
ax_mom.bar(range(len(all_mom_feats)), mom_imps, color=mom_colors)
ax_mom.set_xticks(range(len(all_mom_feats)))
ax_mom.set_xticklabels(all_mom_feats, rotation=35, ha='right', color=P["text"])
ax_mom.set_title("Importance of momentum features", color=P["text"])

plt.tight_layout()
plt.show()

print("\nOptimization v3 completed.\n")