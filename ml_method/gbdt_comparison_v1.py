"""
Comparaison GBDT : XGBoost vs LightGBM vs CatBoost + Stacking
==============================================================
Même feature set (sans leakage), même split chronologique.
Record à battre : R² = 0.2304 | MAE = 0.3533
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
from sklearn.ensemble import StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

RECORD_R2  = 0.2527
RECORD_MAE = 0.3461

print("=" * 65)
print("  Comparaison GBDT : XGBoost vs LightGBM vs CatBoost + Stack")
print("=" * 65)

# ──────────────────────────────────────────────
# 1. CHARGEMENT & FEATURE ENGINEERING
# ──────────────────────────────────────────────
df = pd.read_csv("cleaned_data.csv")

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

df["caption_len"]          = df["caption"].fillna("").str.len()
df["has_emoji"]            = df["caption"].fillna("").apply(lambda s: int(any(ord(c) > 127 for c in s)))
df["has_question"]         = df["caption"].fillna("").str.contains(r"\?").astype(int)
df["has_exclamation"]      = df["caption"].fillna("").str.contains(r"!").astype(int)
df["viral_potential"]      = df["hist_p90_views"] / (df["hist_median_views"] + 1)
df["engagement_total_hist"]= df["hist_like_rate"] + df["hist_comment_rate"] + df["hist_share_rate"]
df["is_peak_hour"]         = df["hour"].between(17, 22).astype(int)
df["follower_tier"]        = np.log1p(df["followers"])
df["views_efficiency_trend"]= df["hist_p70_views"] / (df["hist_median_views"] + 1)

FEATURE_COLS = [
    "followers", "duration", "hour", "weekday", "musicOriginal",
    "hist_median_views", "hist_p70_views", "hist_p90_views",
    "hist_like_rate", "hist_comment_rate", "hist_share_rate",
    "n_hashtags", "has_fyp", "has_viral", "has_foryou",
    "caption_len", "has_emoji", "has_question", "has_exclamation",
    "viral_potential", "engagement_total_hist", "is_peak_hour",
    "follower_tier", "views_efficiency_trend",
]
TARGET = "target_log"

print(f"\n✅ Dataset prêt — {len(FEATURE_COLS)} features | record à battre : R²={RECORD_R2}")

# ──────────────────────────────────────────────
# 2. SPLIT
# ──────────────────────────────────────────────
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

print(f"📊 Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}\n")

# ──────────────────────────────────────────────
# 3. HELPER : tune + evaluate
# ──────────────────────────────────────────────
results = {}

def tune_and_eval(name, estimator, param_dist, n_iter=40, fit_kwargs=None):
    print(f"{'─'*65}")
    print(f"  {name}")
    print(f"{'─'*65}")
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
    print(f"  Meilleurs params : { {k: v for k, v in sorted(best_params.items())} }")

    # Refit sur train seul avec eval_set
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
        callbacks = [
            __import__("lightgbm").early_stopping(100, verbose=False),
            __import__("lightgbm").log_evaluation(-1),
        ]
        final.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=callbacks)
        best_iter = final.best_iteration_
    elif isinstance(final, CatBoostRegressor):
        final.set_params(early_stopping_rounds=100, random_state=42, verbose=0)
        final.fit(X_train, y_train, eval_set=(X_val, y_val))
        best_iter = final.get_best_iteration()

    y_pred = final.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    elapsed = time.time() - t0

    icon_r2  = "🟢" if r2  > RECORD_R2  else "🔴"
    icon_mae = "🟢" if mae < RECORD_MAE else "🔴"

    print(f"  Best iteration   : {best_iter}")
    print(f"  R²  (test)       : {r2:.4f}  {icon_r2}  (Δ {r2 - RECORD_R2:+.4f} vs record)")
    print(f"  MAE (test)       : {mae:.4f}  {icon_mae}  (Δ {mae - RECORD_MAE:+.4f} vs record)")
    print(f"  Temps            : {elapsed:.0f}s\n")

    results[name] = {"model": final, "r2": r2, "mae": mae,
                     "params": best_params, "time": elapsed}
    return final

# ──────────────────────────────────────────────
# 4. XGBOOST (best params connus + exploration)
# ──────────────────────────────────────────────
xgb_params = {
    'n_estimators': [400, 500, 600],
    'learning_rate': [0.02, 0.03, 0.04],
    'max_depth': [2, 3, 4],
    'min_child_weight': [2, 3, 4],
    'reg_lambda': [1, 3, 5],
    'reg_alpha': [0.1, 0.5, 1.0],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

xgb_base = XGBRegressor(tree_method="hist", early_stopping_rounds=100,
                         eval_metric="mae", verbosity=0, random_state=42)
xgb_model = tune_and_eval(
    "XGBoost", xgb_base, xgb_params, n_iter=30,
    fit_kwargs={"eval_set": [(X_val, y_val)]}
)

# ──────────────────────────────────────────────
# 5. LIGHTGBM
# ──────────────────────────────────────────────
import lightgbm as lgb

lgbm_params = {
    "max_depth":         [5, 7, 9, -1],
    "learning_rate":     [0.005, 0.01, 0.03, 0.05],
    "n_estimators":      [2000, 3000, 5000],
    "num_leaves":        [31, 63, 127],          # clé pour LightGBM leaf-wise
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
    fit_kwargs={"eval_set": [(X_val, y_val)],
                "callbacks": callbacks_fit}
)

# ──────────────────────────────────────────────
# 6. CATBOOST
# ──────────────────────────────────────────────
cat_params = {
    "depth":            [4, 6, 8, 10],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "iterations":       [1000, 2000, 3000],
    "l2_leaf_reg":      [1, 3, 5, 10],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bylevel":[0.7, 0.8, 0.9],
    "min_data_in_leaf": [1, 5, 10],
}
cat_base = CatBoostRegressor(early_stopping_rounds=100, random_state=42, verbose=0)
cat_model = tune_and_eval(
    "CatBoost", cat_base, cat_params, n_iter=40,
    fit_kwargs={"eval_set": (X_val, y_val)}
)

# ──────────────────────────────────────────────
# 7. STACKING
# ──────────────────────────────────────────────
print("─" * 65)
print("  Stacking (XGB + LightGBM + CatBoost → Ridge)")
print("─" * 65)
t0 = time.time()

# Refit propre des 3 modèles sur train pour le stacking
xgb_s  = XGBRegressor(**results["XGBoost"]["params"],
                       tree_method="hist", early_stopping_rounds=100,
                       eval_metric="mae", verbosity=0, random_state=42)
lgbm_s = LGBMRegressor(**results["LightGBM"]["params"], random_state=42, verbose=-1)
cat_s  = CatBoostRegressor(**results["CatBoost"]["params"],
                            early_stopping_rounds=100, random_state=42, verbose=0)

# Fit individuels avec eval_set pour early stopping
xgb_s.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
lgbm_s.fit(X_train, y_train, eval_set=[(X_val, y_val)],
           callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
cat_s.fit(X_train, y_train, eval_set=(X_val, y_val))

# Prédictions OOF sur validation → méta-features
meta_val = np.column_stack([
    xgb_s.predict(X_val),
    lgbm_s.predict(X_val),
    cat_s.predict(X_val),
])

# Méta-modèle Ridge entraîné sur val
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_val, y_val)

# Prédictions finales sur test
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
print(f"  Poids méta-modèle : XGB={weights[0]:.2f} | LGBM={weights[1]:.2f} | CAT={weights[2]:.2f}")
print(f"  R²  (test)        : {r2_stack:.4f}  {'🟢' if r2_stack > RECORD_R2 else '🔴'}"
      f"  (Δ {r2_stack - RECORD_R2:+.4f} vs record)")
print(f"  MAE (test)        : {mae_stack:.4f}  {'🟢' if mae_stack < RECORD_MAE else '🔴'}"
      f"  (Δ {mae_stack - RECORD_MAE:+.4f} vs record)")
print(f"  Temps             : {elapsed:.0f}s\n")

results["Stacking"] = {"r2": r2_stack, "mae": mae_stack, "time": elapsed}

# ──────────────────────────────────────────────
# 8. TABLEAU RÉCAPITULATIF
# ──────────────────────────────────────────────
print("=" * 65)
print("  RÉCAPITULATIF FINAL")
print("=" * 65)
print(f"\n{'Modèle':<18} {'R²':>8} {'Δ R²':>8} {'MAE':>8} {'Δ MAE':>8}")
print("─" * 55)
for name, res in results.items():
    dr2  = res["r2"]  - RECORD_R2
    dmae = res["mae"] - RECORD_MAE
    icon = "🏆" if res["r2"] == max(r["r2"] for r in results.values()) else "  "
    print(f"{icon} {name:<16} {res['r2']:>8.4f} {dr2:>+8.4f} {res['mae']:>8.4f} {dmae:>+8.4f}")
print("─" * 55)
print(f"   {'Record XGB':16} {RECORD_R2:>8.4f} {'':>8} {RECORD_MAE:>8.4f}")

best_name = max(results, key=lambda k: results[k]["r2"])
print(f"\n🏆 Meilleur modèle : {best_name}  (R²={results[best_name]['r2']:.4f})")

# ──────────────────────────────────────────────
# 9. VISUALISATION
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle("GBDT Showdown — XGBoost vs LightGBM vs CatBoost vs Stacking",
             fontsize=15, color="white", fontweight="bold", y=1.01)

P = dict(ax_bg="#1a1a1a", grid="#2a2a2a", text="#f0f0f0", gold="#ffbe0b",
         record="#ff4d6d")
COLORS = {
    "XGBoost":  "#3a86ff",
    "LightGBM": "#06d6a0",
    "CatBoost": "#8338ec",
    "Stacking": "#ffbe0b",
}

model_names = list(results.keys())
r2_vals  = [results[n]["r2"]  for n in model_names]
mae_vals = [results[n]["mae"] for n in model_names]
colors   = [COLORS[n] for n in model_names]

for ax, vals, metric, higher_better in [
    (axes[0], r2_vals,  "R²",  True),
    (axes[1], mae_vals, "MAE", False),
]:
    ax.set_facecolor(P["ax_bg"])
    bars = ax.bar(model_names, vals, color=colors, alpha=0.88, zorder=3, width=0.55)

    # Highlight best
    best_idx = (np.argmax(vals) if higher_better else np.argmin(vals))
    bars[best_idx].set_edgecolor(P["gold"])
    bars[best_idx].set_linewidth(2.5)
    bars[best_idx].set_alpha(1.0)

    # Record line
    record_val = RECORD_R2 if metric == "R²" else RECORD_MAE
    ax.axhline(record_val, color=P["record"], linewidth=1.5,
               linestyle="--", zorder=4, label=f"Record XGB ({record_val:.4f})")

    # Value labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                f"{val:.4f}", ha="center", va="bottom",
                color=P["text"], fontsize=10, fontweight="bold")

    ax.set_title(metric, color=P["text"], fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Score", color=P["text"])
    ax.tick_params(colors=P["text"])
    ax.yaxis.grid(True, color=P["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(framealpha=0.2, labelcolor=P["text"], fontsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor(P["grid"])

plt.tight_layout()
out_path = "gbdt_comparison.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n📈 Graphique sauvegardé → {out_path}")
plt.show()
print("\n✅ Comparaison terminée.\n")
