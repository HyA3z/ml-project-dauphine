"""
XGBoost Hyperparameter Tuning — Prédiction de viralité TikTok
=============================================================
Target   : target_log (log de l'Explosion Score)
Split    : Train (rang 11-26) | Validation (27-28) | Test (29-30)
Méthode  : RandomizedSearchCV + early stopping sur le val set
"""

import ast
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. CHARGEMENT
# ──────────────────────────────────────────────
print("=" * 60)
print("  XGBoost TikTok Virality — Hyperparameter Tuning")
print("=" * 60)

df = pd.read_csv("cleaned_data.csv")
print(f"\n✅ Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# ──────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────

# --- Hashtags ---
def parse_hashtags(raw):
    """Parse une liste de dicts [{'name': 'fyp'}, ...] en liste de noms."""
    try:
        tags = ast.literal_eval(raw) if isinstance(raw, str) else raw
        return [t["name"].lower() for t in tags if isinstance(t, dict)]
    except Exception:
        return []

df["_tags"] = df["hashtag"].apply(parse_hashtags)
df["n_hashtags"]   = df["_tags"].apply(len)
df["has_fyp"]      = df["_tags"].apply(lambda t: int(any("fyp" in x for x in t)))
df["has_viral"]    = df["_tags"].apply(lambda t: int(any("viral" in x for x in t)))
df["has_foryou"]   = df["_tags"].apply(lambda t: int(any("foryou" in x for x in t)))
df.drop(columns=["_tags"], inplace=True)

# --- Caption ---
df["caption_len"]      = df["caption"].fillna("").str.len()
df["has_emoji"]        = df["caption"].fillna("").apply(
    lambda s: int(any(ord(c) > 127 for c in s))
)
df["has_question"]     = df["caption"].fillna("").str.contains(r"\?").astype(int)
df["has_exclamation"]  = df["caption"].fillna("").str.contains(r"!").astype(int)

print("\n📐 Features créées :")
print("   Hashtag  → n_hashtags, has_fyp, has_viral, has_foryou")
print("   Caption  → caption_len, has_emoji, has_question, has_exclamation")

# ──────────────────────────────────────────────
# 3. SÉLECTION DES FEATURES
# ──────────────────────────────────────────────
FEATURE_COLS = [
    # Numériques de base
    "followers", "duration", "hour", "weekday", "musicOriginal",
    # Historiques
    "hist_median_views", "hist_p70_views", "hist_p90_views",
    "hist_like_rate", "hist_comment_rate", "hist_share_rate",
    # Hashtag engineered
    "n_hashtags", "has_fyp", "has_viral", "has_foryou",
    # Caption engineered
    "caption_len", "has_emoji", "has_question", "has_exclamation",
]
TARGET = "target_log"

print(f"\n🔢 {len(FEATURE_COLS)} features utilisées")

# ──────────────────────────────────────────────
# 4. SPLIT CHRONOLOGIQUE
# ──────────────────────────────────────────────
train_mask = df["video_rank"].between(11, 26)
val_mask   = df["video_rank"].between(27, 28)
test_mask  = df["video_rank"].between(29, 30)

X_train = df.loc[train_mask, FEATURE_COLS].astype(float)
y_train = df.loc[train_mask, TARGET]

X_val   = df.loc[val_mask, FEATURE_COLS].astype(float)
y_val   = df.loc[val_mask, TARGET]

X_test  = df.loc[test_mask, FEATURE_COLS].astype(float)
y_test  = df.loc[test_mask, TARGET]

# Concat train+val pour RandomizedSearchCV avec PredefinedSplit
X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])

# PredefinedSplit : -1 = train, 0 = validation
split_indices = np.array([-1] * len(X_train) + [0] * len(X_val))
ps = PredefinedSplit(split_indices)

print(f"\n📊 Split chronologique :")
print(f"   Train      (rang 11-26) : {len(X_train):,} lignes")
print(f"   Validation (rang 27-28) : {len(X_val):,} lignes")
print(f"   Test       (rang 29-30) : {len(X_test):,} lignes")

# ──────────────────────────────────────────────
# 5. BASELINE (sans tuning)
# ──────────────────────────────────────────────
print("\n" + "─" * 60)
print("  Étape 1/2 — Baseline (paramètres par défaut)")
print("─" * 60)

baseline = XGBRegressor(
    n_estimators=300,
    random_state=42,
    tree_method="hist",
    verbosity=0,
)
baseline.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

y_pred_base = baseline.predict(X_test)
r2_base  = r2_score(y_test, y_pred_base)
mae_base = mean_absolute_error(y_test, y_pred_base)

print(f"  R²  (test) : {r2_base:.4f}")
print(f"  MAE (test) : {mae_base:.4f}")

# ──────────────────────────────────────────────
# 6. HYPERPARAMETER TUNING
# ──────────────────────────────────────────────
print("\n" + "─" * 60)
print("  Étape 2/2 — RandomizedSearchCV")
print("─" * 60)

param_dist = {
    'n_estimators': [400, 500, 600],
    'learning_rate': [0.02, 0.03, 0.04],
    'max_depth': [2, 3, 4],
    'min_child_weight': [2, 3, 4],
    'reg_lambda': [1, 3, 5],
    'reg_alpha': [0.1, 0.5, 1.0],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

xgb_base = XGBRegressor(
    random_state=42,
    tree_method="hist",
    early_stopping_rounds=100,
    verbosity=0,
    eval_metric="mae",
)

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=100,
    cv=ps,
    scoring="neg_mean_absolute_error",
    refit=False,           # On refit manuellement pour passer eval_set
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("\n🔍 Lancement de RandomizedSearchCV (30 itérations)…\n")
search.fit(
    X_tv, y_tv,
    eval_set=[(X_val, y_val)],
)

best_params = search.best_params_
print(f"\n🏆 Meilleurs paramètres trouvés :")
for k, v in sorted(best_params.items()):
    print(f"   {k:<22}: {v}")

# ──────────────────────────────────────────────
# 7. REFIT DU MEILLEUR MODÈLE
# ──────────────────────────────────────────────
best_model = XGBRegressor(
    **best_params,
    random_state=42,
    tree_method="hist",
    early_stopping_rounds=50,
    verbosity=0,
    eval_metric="mae",
)
best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

y_pred_best = best_model.predict(X_test)
r2_best  = r2_score(y_test, y_pred_best)
mae_best = mean_absolute_error(y_test, y_pred_best)

# ──────────────────────────────────────────────
# 8. COMPARAISON DES PERFORMANCES
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RÉSULTATS SUR LE TEST SET (rang 29-30)")
print("=" * 60)
print(f"\n{'Métrique':<10} {'Baseline':>12} {'Optimisé':>12} {'Δ':>10}")
print("─" * 46)
print(f"{'R²':<10} {r2_base:>12.4f} {r2_best:>12.4f} {r2_best - r2_base:>+10.4f}")
print(f"{'MAE':<10} {mae_base:>12.4f} {mae_best:>12.4f} {mae_best - mae_base:>+10.4f}")
print("─" * 46)

# ──────────────────────────────────────────────
# 9. VISUALISATIONS
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#0f0f0f")
fig.suptitle("XGBoost — TikTok Virality Prediction", fontsize=16,
             color="white", fontweight="bold", y=1.01)

palette = {
    "bar_base":  "#3a86ff",
    "bar_opt":   "#ff006e",
    "fi_bar":    "#8338ec",
    "text":      "#f0f0f0",
    "grid":      "#2a2a2a",
    "ax_bg":     "#1a1a1a",
}

# --- Graphique 1 : Comparaison Baseline vs Optimisé ---
ax1 = axes[0]
ax1.set_facecolor(palette["ax_bg"])

metrics   = ["R²", "MAE"]
baseline_scores = [r2_base, mae_base]
tuned_scores    = [r2_best, mae_best]
x = np.arange(len(metrics))
w = 0.35

bars1 = ax1.bar(x - w/2, baseline_scores, w, label="Baseline",
                color=palette["bar_base"], alpha=0.9, zorder=3)
bars2 = ax1.bar(x + w/2, tuned_scores,    w, label="Optimisé",
                color=palette["bar_opt"],  alpha=0.9, zorder=3)

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., h + 0.003,
             f"{h:.4f}", ha="center", va="bottom",
             color=palette["text"], fontsize=9, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(metrics, color=palette["text"], fontsize=12)
ax1.set_title("Baseline vs Optimisé (Test Set)", color=palette["text"],
              fontsize=13, fontweight="bold", pad=12)
ax1.set_ylabel("Score", color=palette["text"])
ax1.tick_params(colors=palette["text"])
ax1.legend(framealpha=0.2, labelcolor=palette["text"])
ax1.yaxis.grid(True, color=palette["grid"], linewidth=0.8, zorder=0)
ax1.set_axisbelow(True)
for spine in ax1.spines.values():
    spine.set_edgecolor(palette["grid"])

# --- Graphique 2 : Feature Importance ---
ax2 = axes[1]
ax2.set_facecolor(palette["ax_bg"])

fi = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
fi_sorted = fi.sort_values(ascending=True)

colors_fi = [palette["fi_bar"]] * len(fi_sorted)
bars_fi = ax2.barh(fi_sorted.index, fi_sorted.values,
                   color=colors_fi, alpha=0.85, zorder=3)

# Highlight top 3
for i, (idx, bar) in enumerate(zip(fi_sorted.index[::-1], bars_fi[::-1])):
    if i < 3:
        bar.set_color("#ffbe0b")
        bar.set_alpha(1.0)

ax2.set_title("Feature Importance (meilleur modèle)", color=palette["text"],
              fontsize=13, fontweight="bold", pad=12)
ax2.set_xlabel("Importance", color=palette["text"])
ax2.tick_params(colors=palette["text"], axis="y", labelsize=9)
ax2.tick_params(colors=palette["text"], axis="x")
ax2.xaxis.grid(True, color=palette["grid"], linewidth=0.8, zorder=0)
ax2.set_axisbelow(True)
for spine in ax2.spines.values():
    spine.set_edgecolor(palette["grid"])

# Légende couleur
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor="#ffbe0b", label="Top 3 features"),
    Patch(facecolor=palette["fi_bar"], alpha=0.85, label="Autres features"),
]
ax2.legend(handles=legend_els, framealpha=0.2, labelcolor=palette["text"], fontsize=9)

plt.tight_layout()
out_path = "feature_importance_xgboost.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\n📈 Graphique sauvegardé → {out_path}")
plt.show()

# ──────────────────────────────────────────────
# 10. TOP FEATURES
# ──────────────────────────────────────────────
print("\n🔑 Top 5 features les plus importantes :")
top5 = fi.sort_values(ascending=False).head(5)
for feat, score in top5.items():
    print(f"   {feat:<25} {score:.4f}")

print("\n✅ Script terminé avec succès.\n")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Assuming best_model and FEATURE_COLS are defined as in your script

# ──────────────────────────────────────────────
# 9. GENERATE CLEAN FEATURE IMPORTANCE PLOT (WHITE BACKGROUND)
# ──────────────────────────────────────────────

# 1. Base Setup for a clean, professional, white plot
plt.style.use('default')  # Start with standard white
sns.set_style("whitegrid", {
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "grid.color": "#e0e0e0", # Light grid
    "grid.linestyle": "-",
    "axes.edgecolor": "#333333", # Clean dark gray edges
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#333333"
})

# 2. Extract and Sort Importance
fi = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
fi_sorted = fi.sort_values(ascending=True)

# 3. Define Clean, Distinct Colors (no transparency, professional palette)
# Main color: A muted, professional Teal/Blue
main_color = "#4c72b0" # A classic Seaborn deep blue, solid
# Highlight color for Top 3: A rich, warm Orange/Gold
highlight_color = "#e08214" # Solid, distinct orange

# Initialize color list
colors = [main_color] * len(fi_sorted)
# Apply highlight to the top 3 (highest importance)
for i in range(1, 4):
    colors[-i] = highlight_color

# 4. Initialize Plot
fig, ax = plt.subplots(figsize=(12, 8))
# Ensure figure background is explicitly white for saving
fig.patch.set_facecolor('white')

# 5. Create the horizontal bar plot (NO transparency 'alpha=0.9', fully solid)
bars = ax.barh(fi_sorted.index, fi_sorted.values, 
                color=colors, 
                edgecolor="white", # Thin white edge separates bars cleanly
                linewidth=1,
                zorder=3) # Ensure bars are above the grid

# 6. Add Score Labels to the end of each bar (Black text, no transparency)
for bar in bars:
    width = bar.get_width()
    # Adjust position slightly based on width for visibility
    label_x_pos = width + 0.005 
    ax.text(label_x_pos, 
            bar.get_y() + bar.get_height() / 2, 
            f'{width:.4f}', 
            va='center', 
            ha='left', # Left-align label from the point
            color='#111111', # True black text for maximum contrast
            fontsize=10, 
            fontweight='bold', 
            zorder=4) # Labels above bars

# 7. Customize Axes for a clean look (English Titles)
# Main title
ax.set_title("XGBoost Feature Importance — TikTok Virality Prediction\n(Optimized Model $R^2 = 0.2527$)", 
             fontsize=16, fontweight='bold', pad=25, color="#111111")

# X-axis label
ax.set_xlabel("Importance Score (Gain Metric)", fontsize=13, fontweight='bold', color="#333333", labelpad=15)

# Clean up axes: Remove top/right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Make left/bottom spines solid gray
ax.spines['left'].set_color('#333333')
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_color('#333333')
ax.spines['bottom'].set_linewidth(1.2)

# Grid: Only vertical lines, light gray
ax.xaxis.grid(True, linestyle='-', color="#e0e0e0", zorder=0)
ax.yaxis.grid(False) # No horizontal grid lines

# Tick parameters: clean black/gray
ax.tick_params(axis='both', colors='#333333', labelsize=11)
ax.tick_params(axis='x', which='major', pad=8) # Add space for x-labels

# 8. Legend for clarity (using solid patches)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=highlight_color, edgecolor='white', label='Top 3 Predictors'),
    Patch(facecolor=main_color, edgecolor='white', label='Other Predictors')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
          facecolor='white', edgecolor='#e0e0e0', fontsize=11)

# 9. Fine-tune layout and SAVE (English Filename)
plt.tight_layout()
# dpi=300 for high resolution, transparent=False to ensure white background is solid
plt.savefig("tiktok_xgboost_feature_importance.png", dpi=300, bbox_inches="tight", facecolor='white', transparent=False)

print("\n✅ High-resolution, clean chart (tiktok_xgboost_feature_importance.png) generated on SOLID WHITE background.")
plt.show()