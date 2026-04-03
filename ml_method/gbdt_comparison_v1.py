import ast
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
RECORD_R2 = 0.2527
RECORD_MAE = 0.3461
TARGET = "target_log"
FEATURE_COLS = [
    "followers", "duration", "hour", "weekday", "musicOriginal",
    "hist_median_views", "hist_p70_views", "hist_p90_views",
    "hist_like_rate", "hist_comment_rate", "hist_share_rate",
    "n_hashtags", "has_fyp", "has_viral", "has_foryou",
    "caption_len", "has_emoji", "has_question", "has_exclamation",
    "viral_potential", "engagement_total_hist", "is_peak_hour",
    "follower_tier", "views_efficiency_trend",
]

# ---------------------------------------------------------
# 1. DATA LOADING & FEATURE ENGINEERING
# ---------------------------------------------------------
def parse_hashtags(raw):
    try:
        tags = ast.literal_eval(raw) if isinstance(raw, str) else raw
        return [t["name"].lower() for t in tags if isinstance(t, dict)]
    except (ValueError, SyntaxError, TypeError):
        return []

def prepare_data(path="cleaned_data.csv"):
    df = pd.read_csv(path)
    
    # Hashtag Features
    df["_tags"] = df["hashtag"].apply(parse_hashtags)
    df["n_hashtags"] = df["_tags"].apply(len)
    df["has_fyp"] = df["_tags"].apply(lambda t: int(any("fyp" in x for x in t)))
    df["has_viral"] = df["_tags"].apply(lambda t: int(any("viral" in x for x in t)))
    df["has_foryou"] = df["_tags"].apply(lambda t: int(any("foryou" in x for x in t)))
    
    # Caption & Engagement Features
    df["caption_len"] = df["caption"].fillna("").str.len()
    df["has_emoji"] = df["caption"].fillna("").apply(lambda s: int(any(ord(c) > 127 for c in s)))
    df["has_question"] = df["caption"].fillna("").str.contains(r"\?").astype(int)
    df["has_exclamation"] = df["caption"].fillna("").str.contains(r"!").astype(int)
    
    df["viral_potential"] = df["hist_p90_views"] / (df["hist_median_views"] + 1)
    df["engagement_total_hist"] = df["hist_like_rate"] + df["hist_comment_rate"] + df["hist_share_rate"]
    df["is_peak_hour"] = df["hour"].between(17, 22).astype(int)
    df["follower_tier"] = np.log1p(df["followers"])
    df["views_efficiency_trend"] = df["hist_p70_views"] / (df["hist_median_views"] + 1)
    
    return df.drop(columns=["_tags"])

df = prepare_data()

# Chronological Split Based on Video Rank
train_mask = df["video_rank"].between(11, 26)
val_mask = df["video_rank"].between(27, 28)
test_mask = df["video_rank"].between(29, 30)

X_train, y_train = df.loc[train_mask, FEATURE_COLS].astype(float), df.loc[train_mask, TARGET]
X_val, y_val = df.loc[val_mask, FEATURE_COLS].astype(float), df.loc[val_mask, TARGET]
X_test, y_test = df.loc[test_mask, FEATURE_COLS].astype(float), df.loc[test_mask, TARGET]

# Combined Train+Val for CV
X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])
ps = PredefinedSplit(np.array([-1] * len(X_train) + [0] * len(X_val)))

print(f"Dataset ready: {len(X_train)} Train | {len(X_val)} Val | {len(X_test)} Test")

# ---------------------------------------------------------
# 2. BENCHMARKING ENGINE
# ---------------------------------------------------------
class GBDTBenchmarker:
    def __init__(self):
        self.results = {}

    def run(self, name, estimator, param_dist, n_iter=30):
        print(f"\n{'#'*10} Tuning {name} {'#'*10}")
        start_time = time.time()
        
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=ps,
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_tv, y_tv)
        best_params = search.best_params_
        
        # Initialize final model with best params
        model = estimator.__class__(**{**estimator.get_params(), **best_params})
        
        # Specific Early Stopping Logic
        if isinstance(model, XGBRegressor):
            model.set_params(early_stopping_rounds=100, eval_metric="mae")
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            best_iteration = model.best_iteration
        elif isinstance(model, LGBMRegressor):
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      callbacks=[early_stopping(100, verbose=False), log_evaluation(-1)])
            best_iteration = model.best_iteration_
        elif isinstance(model, CatBoostRegressor):
            model.set_params(early_stopping_rounds=100, verbose=0)
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            best_iteration = model.get_best_iteration()

        # Evaluation
        y_pred = model.predict(X_test)
        r2, mae = r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)
        elapsed = time.time() - start_time
        
        self.results[name] = {
            "model": model, "r2": r2, "mae": mae, 
            "params": best_params, "time": elapsed
        }
        
        print(f"Best Iteration: {best_iteration}")
        print(f"R²: {r2:.4f} ({'🟢' if r2 > RECORD_R2 else '🔴'})")
        print(f"MAE: {mae:.4f} ({'🟢' if mae < RECORD_MAE else '🔴'})")
        return model

bench = GBDTBenchmarker()

# ---------------------------------------------------------
# 3. MODEL TRAINING
# ---------------------------------------------------------

# XGBoost
xgb_params = {
    'n_estimators': [500, 600], 'learning_rate': [0.02, 0.03],
    'max_depth': [3, 4], 'subsample': [0.8, 0.9]
}
bench.run("XGBoost", XGBRegressor(tree_method="hist", random_state=42), xgb_params)

# LightGBM
lgbm_params = {
    "num_leaves": [31, 63], "learning_rate": [0.01, 0.03],
    "n_estimators": [3000, 5000], "min_child_samples": [20, 50]
}
bench.run("LightGBM", LGBMRegressor(random_state=42, verbose=-1), lgbm_params)

# CatBoost
cat_params = {
    "depth": [6, 8], "learning_rate": [0.03, 0.05], 
    "iterations": [2000], "l2_leaf_reg": [3, 5]
}
bench.run("CatBoost", CatBoostRegressor(random_state=42, verbose=0), cat_params)

# ---------------------------------------------------------
# 4. STACKING (Meta-Learner)
# ---------------------------------------------------------
print(f"\n{'#'*10} Stacking Ensemble {'#'*10}")
t_start = time.time()

# Extract models
xgb_m, lgbm_m, cat_m = bench.results["XGBoost"]["model"], bench.results["LightGBM"]["model"], bench.results["CatBoost"]["model"]

# Meta-features from Validation set
meta_val = np.column_stack([xgb_m.predict(X_val), lgbm_m.predict(X_val), cat_m.predict(X_val)])
meta_model = Ridge(alpha=1.0).fit(meta_val, y_val)

# Final Prediction on Test
meta_test = np.column_stack([xgb_m.predict(X_test), lgbm_m.predict(X_test), cat_m.predict(X_test)])
y_pred_stack = meta_model.predict(meta_test)

r2_stack = r2_score(y_test, y_pred_stack)
mae_stack = mean_absolute_error(y_test, y_pred_stack)

bench.results["Stacking"] = {"r2": r2_stack, "mae": mae_stack, "time": time.time() - t_start}

# ---------------------------------------------------------
# 5. SUMMARY & VISUALIZATION
# ---------------------------------------------------------
print("\n" + "="*65)
print(f"{'Model':<15} | {'R2':>8} | {'Delta R2':>10} | {'MAE':>8}")
print("-" * 65)
for name, res in bench.results.items():
    d_r2 = res['r2'] - RECORD_R2
    print(f"{name:<15} | {res['r2']:>8.4f} | {d_r2:>+10.4f} | {res['mae']:>8.4f}")

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
models = list(bench.results.keys())
r2_scores = [bench.results[m]['r2'] for m in models]
mae_scores = [bench.results[m]['mae'] for m in models]

ax1.bar(models, r2_scores, color=['#3498db', '#2ecc71', '#9b59b6', '#f1c40f'])
ax1.axhline(RECORD_R2, color='red', linestyle='--', label='Record')
ax1.set_title("R² Comparison (Higher is Better)")
ax1.legend()

ax2.bar(models, mae_scores, color=['#3498db', '#2ecc71', '#9b59b6', '#f1c40f'])
ax2.axhline(RECORD_MAE, color='red', linestyle='--', label='Record')
ax2.set_title("MAE Comparison (Lower is Better)")
ax2.legend()

plt.tight_layout()
plt.show()