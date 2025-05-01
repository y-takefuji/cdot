import pandas as pd
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score

# PARAMETERS
TEST_SIZE    = 0.20
RANDOM_STATE = 42
TOPK         = 100

# 1) LOAD & DROP UNUSED COLUMNS
df = pd.read_csv("cdot_data.csv")
df.drop(columns=[
    "project_number",
    "engineers_estimate",
    "bid_days",
    "start_date"
], inplace=True)

# 2) EXTRACT CONTINUOUS TARGET (no binning)
y = df.pop("bid_total").values
X = df.values
feature_names = df.columns

# 3) RANK FEATURES USING RandomForestRegressor IMPORTANCES
rf_fs = RandomForestRegressor(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_fs.fit(X, y)

importances = pd.Series(rf_fs.feature_importances_, index=feature_names)
top_feats  = importances.nlargest(TOPK).index.tolist()

print(f"Top {TOPK} features by importance:\n", importances.nlargest(TOPK))

# 4) REDUCE TO TOP-K AND SPLIT (no stratify for regression)
X_top = df[top_feats].values

# 5) 10-FOLD CV ON TRAINING SET WITH A REGRESSOR
rf_reg = RandomForestRegressor(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
# you can choose scoring='r2' or scoring='neg_mean_squared_error', etc.
cv_scores_r2 = cross_val_score(
    rf_reg, X_top, y,
    cv=10,
    scoring="r2",
    n_jobs=-1
)

print(f"\nCV R²: mean = {cv_scores_r2.mean():.3f} ± {cv_scores_r2.std():.3f}")

