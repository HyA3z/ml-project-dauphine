"""
Stacking + Dynamic Momentum — Final TikTok Virality Optimization
==================================================================
Updates vs previous version:
  - Dynamic Feature Engineering: momentum_3, trend_slope, consistency
  - Reconstruction of creator_id via groupby(followers)
  - Strict shift(1) to prevent data leakage
Record to beat: R² = 0.3018 | MAE = 0.3030
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
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

RECORD_R2  = 0.3018
RECORD_MAE = 0.3030

print("=" * 70)
print("  Stacking + Dynamic Momentum — TikTok Virality Prediction")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# 1. LOADING
# ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("cleaned_data.csv")

# ─────────────────────────────────────────────────────────────────────
# 2. CREATOR_ID RECONSTRUCTION
#    Assumption: each creator has a unique follower count.
#    We reconstruct a stable ID by de-duplicating by (followers).
# ─────────────────────────────────────────────────────────────────────
# Chronological sort to guarantee the order of shifts
df = df.sort_values(["followers", "video_rank"]).reset_index(drop=True)

# Unique numeric identifier per creator
df["creator_id"] = df.groupby("followers").ngroup()

n_creators = df["creator_id"].nunique()
print(f"\n{n_creators} creators identified via 'followers'")

# ─────────────────────────────────────────────────────────────────────
# 3. STATIC FEATURE ENGINEERING (identical to the previous version)
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

df["caption_len"]           = df["caption"].fillna("").str.len()
df["has_emoji"]             = df["caption"].fillna("").apply(lambda s: int(any(ord(c) > 127 for c in s)))
df["has_question"]          = df["caption"].fillna("").str.contains(r"\?").astype(int)
df["has_exclamation"]       = df["caption"].fillna("").str.contains(r"!").astype(int)
df["viral_potential"]       = df["hist_p90_views"] / (df["hist_median_views"] + 1)
df["engagement_total_hist"] = df["hist_like_rate"] + df["hist_comment_rate"] + df["hist_share_rate"]
df["is_peak_hour"]          = df["hour"].between(17, 22).astype(int)
df["follower_tier"]         = np.log1p(df["followers"])
df["views_efficiency_trend"]= df["hist_p70_views"] / (df["hist_median_views"] + 1)

# ─────────────────────────────────────────────────────────────────────
# 4. DYNAMIC FEATURE ENGINEERING — MOMENTUM (no leakage)
#    All features use .shift(1): we only look at
#    PREVIOUS videos, never the current video.
# ─────────────────────────────────────────────────────────────────────
print("\nCalculating Momentum features...")

def rolling_slope(series, window=5):
    """Linear slope (polyfit degree 1) over a rolling window."""
    def slope(arr):
        if arr.isna().any() or len(arr) < 2:
            return np.nan
        x = np.arange(len(arr))
        return np.polyfit(x, arr.values, 1)[0]
    return series.rolling(window, min_periods=2).apply(slope, raw=False)

grp = df.groupby("creator_id")["explosion_score"]

# Shift(1) applied before each rolling window to prevent leakage
shifted = grp.shift(1)

# momentum_3 : moving average over the last 3 videos (after shift)
df["momentum_3"] = (
    shifted
    .groupby(df["creator_id"])
    .transform(lambda s: s.rolling(3, min_periods=1).mean())
)

# trend_slope : slope over the last 5 videos (after shift)
df["trend_slope"] = (
    shifted
    .groupby(df["creator_id"])
    .transform(lambda s: rolling_slope(s, window=5))
)

# consistency : standard deviation over the last 5 videos (after shift)
df["consistency"] = (
    shifted
    .groupby(df["creator_id"])
    .transform(lambda s: s.rolling(5, min_periods=2).std())
)

# Bonus momentum features
# momentum_ratio : momentum_3 normalized by historical median
df["momentum_ratio"] = df["momentum_3"] / (df["hist_median_views"] + 1)

# trend_direction : 1=up, -1=down, 0=stable
df["trend_direction"] = np.sign(df["trend_slope"].fillna(0)).astype(int)

# volatility_tier : consistency relative to the median
df["volatility_tier"] = df["consistency"] / (df["hist_median_views"] + 1)

momentum_features = [
    "momentum_3", "trend_slope", "consistency",
    "momentum_ratio", "trend_direction", "volatility_tier",
]

# Filling momentum NaNs (videos at the beginning of the sequence)
df[momentum_features] = df[momentum_features].fillna(df[momentum_features].median())

n_momentum_nan = df[momentum_features].isna().sum().sum()
print(f"  Momentum features created | Residual NaNs: {n_momentum_nan}")
print(f"  Preview momentum_3 (non-null):")
sample = df[df["momentum_3"] > 0][["creator_id", "video_rank", "explosion_score", "momentum_3", "trend_slope", "consistency"]].head(5)
print(sample.to_string(index=False))

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

FEATURE_COLS = STATIC_FEATURES + momentum_features
TARGET = "target_log"

print(f"\nFeature set: {len(STATIC_FEATURES)} static + {len(momentum_features)} momentum = {len(FEATURE_COLS)} total")

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

print(f"Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")
print(f"   Record to beat: R²={RECORD_R2} | MAE={RECORD_MAE}\n")

# ─────────────────────────────────────────────────────────────────────
# 7. HELPER: tune + evaluate
# ─────────────────────────────────────────────────────────────────────
results = {}

def tune_and_eval(name, estimator, param_dist, n_iter=40, fit_kwargs=None):
    print(f"{'─'*70}")
    print(f"  {name}")
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

    # Refit on train only with early stopping on val
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

    icon_r2  = "[+]" if r2  > RECORD_R2  else "[-]"
    icon_mae = "[+]" if mae < RECORD_MAE else "[-]"

    print(f"  Best iteration   : {best_iter}")
    print(f"  R²  (test)       : {r2:.4f}  {icon_r2}  (Δ {r2 - RECORD_R2:+.4f} vs record)")
    print(f"  MAE (test)       : {mae:.4f}  {icon_mae}  (Δ {mae - RECORD_MAE:+.4f} vs record)")
    print(f"  Time             : {elapsed:.0f}s\n")

    results[name] = {"model": final, "r2": r2, "mae": mae,
                     "params": best_params, "time": elapsed}
    return final

# ─────────────────────────────────────────────────────────────────────
# 8. XGBOOST
# ─────────────────────────────────────────────────────────────────────
xgb_params = {
    "max_depth":        [5, 7, 9],
    "learning_rate":    [0.01, 0.03, 0.05],
    "n_estimators":     [2000, 3000],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3],
    "reg_alpha":        [0.1, 0.5, 1.0],
    "reg_lambda":       [1.0, 2.0],
}
xgb_base = XGBRegressor(tree_method="hist", early_stopping_rounds=100,
                         eval_metric="mae", verbosity=0, random_state=42)
xgb_model = tune_and_eval(
    "XGBoost", xgb_base, xgb_params, n_iter=30,
    fit_kwargs={"eval_set": [(X_val, y_val)]}
)

# ─────────────────────────────────────────────────────────────────────
# 9. LIGHTGBM
# ─────────────────────────────────────────────────────────────────────
lgbm_params = {
    "max_depth":         [5, 7, 9, -1],
    "learning_rate":     [0.005, 0.01, 0.03, 0.05],
    "n_estimators":      [2000, 3000, 5000],
    "num_leaves":        [31, 63, 127],
    "subsample":         [0.7, 0.8, 0.9],
    "colsample_bytree":  [0.7, 0.8, 0.9],
    "min_child_samples": [10, 20, 50],
    "reg_alpha":         [0, 0.1, 0.5],
    "reg_lambda":        [0.5, 1.0, 2.0],
}
lgbm_base = LGBMRegressor(random_state=42, verbose=-1)
callbacks_fit = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
lgbm_model = tune_and_eval(
    "LightGBM", lgbm_base, lgbm_params, n_iter=40,
    fit_kwargs={"eval_set": [(X_val, y_val)], "callbacks": callbacks_fit}
)

# ─────────────────────────────────────────────────────────────────────
# 10. CATBOOST
# ─────────────────────────────────────────────────────────────────────
cat_params = {
    "depth":             [4, 6, 8, 10],
    "learning_rate":     [0.01, 0.03, 0.05, 0.1],
    "iterations":        [1000, 2000, 3000],
    "l2_leaf_reg":       [1, 3, 5, 10],
    "subsample":         [0.7, 0.8, 0.9],
    "colsample_bylevel": [0.7, 0.8, 0.9],
    "min_data_in_leaf":  [1, 5, 10],
}
cat_base = CatBoostRegressor(early_stopping_rounds=100, random_state=42, verbose=0)
cat_model = tune_and_eval(
    "CatBoost", cat_base, cat_params, n_iter=40,
    fit_kwargs={"eval_set": (X_val, y_val)}
)

# ─────────────────────────────────────────────────────────────────────
# 11. STACKING (XGB + LightGBM + CatBoost -> Ridge meta-learner)
# ─────────────────────────────────────────────────────────────────────
print("─" * 70)
print("  Stacking (XGB + LightGBM + CatBoost -> Ridge meta-learner)")
print("─" * 70)
t0 = time.time()

xgb_s = XGBRegressor(**results["XGBoost"]["params"],
                      tree_method="hist", early_stopping_rounds=100,
                      eval_metric="mae", verbosity=0, random_state=42)
lgbm_s = LGBMRegressor(**results["LightGBM"]["params"], random_state=42, verbose=-1)
cat_s  = CatBoostRegressor(**results["CatBoost"]["params"],
                            early_stopping_rounds=100, random_state=42, verbose=0)

xgb_s.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
lgbm_s.fit(X_train, y_train, eval_set=[(X_val, y_val)],
           callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
cat_s.fit(X_train, y_train, eval_set=(X_val, y_val))

# Meta-features on val
meta_val = np.column_stack([
    xgb_s.predict(X_val),
    lgbm_s.predict(X_val),
    cat_s.predict(X_val),
])

# Ridge meta-model trained on val
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_val, y_val)

# Final predictions on test
meta_test = np.column_stack([
    xgb_s.predict(X_test),
    lgbm_s.predict(X_test),
    cat_s.predict(X_test),
])
y_pred_stack = meta_model.predict(meta_test)

r2_stack  = r2_score(y_test, y_pred_stack)
mae_stack = mean_absolute_error(y_test, y_pred_stack)
elapsed   = time.time() - t0

weights = meta_model.coef_ / meta_model.coef_.sum()
print(f"  Meta-model weights : XGB={weights[0]:.2%} | LGBM={weights[1]:.2%} | CAT={weights[2]:.2%}")
print(f"  R²  (test)         : {r2_stack:.4f}  {'[+]' if r2_stack > RECORD_R2 else '[-]'}"
      f"  (Δ {r2_stack - RECORD_R2:+.4f} vs record)")
print(f"  MAE (test)         : {mae_stack:.4f}  {'[+]' if mae_stack < RECORD_MAE else '[-]'}"
      f"  (Δ {mae_stack - RECORD_MAE:+.4f} vs record)")
print(f"  Time               : {elapsed:.0f}s\n")

results["Stacking"] = {"r2": r2_stack, "mae": mae_stack, "time": elapsed}

# ─────────────────────────────────────────────────────────────────────
# 12. MOMENTUM FEATURE IMPORTANCE (via CatBoost)
# ─────────────────────────────────────────────────────────────────────
print("─" * 70)
print("  Momentum feature importance (CatBoost)")
print("─" * 70)

fi = pd.Series(cat_s.get_feature_importance(), index=FEATURE_COLS).sort_values(ascending=False)
print("\n  Top 15 features :")
for feat, imp in fi.head(15).items():
    marker = " <- MOMENTUM" if feat in momentum_features else ""
    print(f"    {feat:<35} {imp:>6.2f}{marker}")

# ─────────────────────────────────────────────────────────────────────
# 13. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"\n{'Model':<18} {'R²':>8} {'Δ R²':>9} {'MAE':>8} {'Δ MAE':>9}")
print("─" * 60)
for name, res in results.items():
    dr2  = res["r2"]  - RECORD_R2
    dmae = res["mae"] - RECORD_MAE
    icon = "*" if res["r2"] == max(r["r2"] for r in results.values()) else " "
    print(f"{icon} {name:<16} {res['r2']:>8.4f} {dr2:>+9.4f} {res['mae']:>8.4f} {dmae:>+9.4f}")
print("─" * 60)
print(f"   {'Record':16} {RECORD_R2:>8.4f} {'':>9} {RECORD_MAE:>8.4f}")

best_name = max(results, key=lambda k: results[k]["r2"])
print(f"\nBest model: {best_name}  (R²={results[best_name]['r2']:.4f})")

# ─────────────────────────────────────────────────────────────────────
# 14. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle("Stacking + Dynamic Momentum — TikTok Virality Prediction",
             fontsize=15, color="white", fontweight="bold", y=1.01)

P = dict(ax_bg="#1a1a1a", grid="#2a2a2a", text="#f0f0f0",
         gold="#ffbe0b", record="#ff4d6d", momentum="#00f5d4")
COLORS = {
    "XGBoost":  "#3a86ff",
    "LightGBM": "#06d6a0",
    "CatBoost": "#8338ec",
    "Stacking": "#ffbe0b",
}

model_names = list(results.keys())
r2_vals     = [results[n]["r2"]  for n in model_names]
mae_vals    = [results[n]["mae"] for n in model_names]
colors      = [COLORS[n] for n in model_names]

# — Axis 1: R²
ax = axes[0]
ax.set_facecolor(P["ax_bg"])
bars = ax.bar(model_names, r2_vals, color=colors, alpha=0.88, zorder=3, width=0.55)
best_idx = np.argmax(r2_vals)
bars[best_idx].set_edgecolor(P["gold"]); bars[best_idx].set_linewidth(2.5); bars[best_idx].set_alpha(1.0)
ax.axhline(RECORD_R2, color=P["record"], linewidth=1.5, linestyle="--", zorder=4,
           label=f"Record ({RECORD_R2:.4f})")
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f"{val:.4f}",
            ha="center", va="bottom", color=P["text"], fontsize=10, fontweight="bold")
ax.set_title("R² Score", color=P["text"], fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Score", color=P["text"])
ax.tick_params(colors=P["text"])
ax.yaxis.grid(True, color=P["grid"], linewidth=0.8, zorder=0)
ax.legend(framealpha=0.2, labelcolor=P["text"], fontsize=9)
for sp in ax.spines.values(): sp.set_edgecolor(P["grid"])

# — Axis 2: MAE
ax = axes[1]
ax.set_facecolor(P["ax_bg"])
bars = ax.bar(model_names, mae_vals, color=colors, alpha=0.88, zorder=3, width=0.55)
best_idx = np.argmin(mae_vals)
bars[best_idx].set_edgecolor(P["gold"]); bars[best_idx].set_linewidth(2.5); bars[best_idx].set_alpha(1.0)
ax.axhline(RECORD_MAE, color=P["record"], linewidth=1.5, linestyle="--", zorder=4,
           label=f"Record ({RECORD_MAE:.4f})")
for bar, val in zip(bars, mae_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002, f"{val:.4f}",
            ha="center", va="bottom", color=P["text"], fontsize=10, fontweight="bold")
ax.set_title("MAE", color=P["text"], fontsize=13, fontweight="bold", pad=12)
ax.tick_params(colors=P["text"])
ax.yaxis.grid(True, color=P["grid"], linewidth=0.8, zorder=0)
ax.legend(framealpha=0.2, labelcolor=P["text"], fontsize=9)
for sp in ax.spines.values(): sp.set_edgecolor(P["grid"])

# — Axis 3: Feature Importance top 15 (momentum highlighted)
ax = axes[2]
ax.set_facecolor(P["ax_bg"])
top15 = fi.head(15)
feat_colors = [P["momentum"] if f in momentum_features else "#aaaaaa" for f in top15.index]
bars_h = ax.barh(range(len(top15)), top15.values, color=feat_colors, alpha=0.88, zorder=3)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index, fontsize=8, color=P["text"])
ax.invert_yaxis()
ax.set_title("Feature Importance (CatBoost)\nBlue=Momentum  Gray=Static",
             color=P["text"], fontsize=11, fontweight="bold", pad=12)
ax.tick_params(colors=P["text"])
ax.xaxis.grid(True, color=P["grid"], linewidth=0.8, zorder=0)
for sp in ax.spines.values(): sp.set_edgecolor(P["grid"])

momentum_patch = mpatches.Patch(color=P["momentum"], label="Momentum feature")
static_patch   = mpatches.Patch(color="#aaaaaa",      label="Static feature")
ax.legend(handles=[momentum_patch, static_patch], framealpha=0.2,
          labelcolor=P["text"], fontsize=9, loc="lower right")

plt.tight_layout()
out_path = "stacking_momentum.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nPlot saved -> {out_path}")
plt.show()
print("\nOptimization completed.\n")