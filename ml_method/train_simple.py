import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import loguniform
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('../dataset/dataset_engineered.csv')

features = [
    'followers', 'duration', 'musicOriginal', 'hour', 'weekday',
    'hist_median_views', 'hist_p70_views', 'hist_p90_views', 
    'hist_like_rate', 'hist_comment_rate', 'hist_share_rate',
    'n_hashtags', 'has_fyp', 'has_viral', 'has_foryou', 
    'caption_len', 'has_emoji', 'has_question', 'has_exclamation',
]
target = 'target_log'

train_val_df = df[df['video_rank'].between(11, 28)].copy()
test_df = df[df['video_rank'].isin([29, 30])].copy()

X_train_val = train_val_df[features]
y_train_val = train_val_df[target]
X_test = test_df[features]
y_test = test_df[target]

test_fold = np.where(train_val_df['video_rank'] <= 26, -1, 0)
ps = PredefinedSplit(test_fold)

scaler = StandardScaler()
X_train_val_scaled = scaler.fit_transform(X_train_val)
X_test_scaled = scaler.transform(X_test)

def evaluate_model(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"[{name}] Test R2: {r2:.4f}, Test MAE: {mae:.4f}")
    return r2, mae


lr = LinearRegression()
lr.fit(X_train_val_scaled[test_fold == -1], y_train_val[test_fold == -1])
evaluate_model("Linear Regression (OLS)", y_test, lr.predict(X_test_scaled))

ridge_param_dist = {
    'alpha': loguniform(1e-3, 1e3),
    'fit_intercept': [True, False]
}
ridge_search = RandomizedSearchCV(
    Ridge(), param_distributions=ridge_param_dist, n_iter=50, 
    cv=ps, scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
)
ridge_search.fit(X_train_val_scaled, y_train_val)
evaluate_model("Ridge (Tuned)", y_test, ridge_search.predict(X_test_scaled))
print(f"Best Ridge Alpha: {ridge_search.best_params_['alpha']:.4f}")

knn_param_dist = {
    'n_neighbors': np.arange(1, 51),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_search = RandomizedSearchCV(
    KNeighborsRegressor(), param_distributions=knn_param_dist, n_iter=30, 
    cv=ps, scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
)
knn_search.fit(X_train_val_scaled, y_train_val)
evaluate_model("KNN (Tuned)", y_test, knn_search.predict(X_test_scaled))
print(f"Best KNN Params: {knn_search.best_params_}")

rf_param_dist = {
    'n_estimators': [100, 200, 500],        
    'max_depth': [None, 10, 20, 30],        
    'min_samples_split': [2, 5, 10],       
    'min_samples_leaf': [1, 4, 10],      
    'max_features': ['sqrt', 'log2', None]  
}

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42), 
    param_distributions=rf_param_dist, 
    n_iter=20,
    cv=ps, 
    scoring='neg_mean_absolute_error', 
    random_state=42, 
    n_jobs=-1
)

rf_search.fit(X_train_val_scaled, y_train_val)

evaluate_model("Random Forest (Tuned)", y_test, rf_search.predict(X_test_scaled))
print(f"Best RF Params: {rf_search.best_params_}")