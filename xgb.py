import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

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

# 3) RANK FEATURES USING XGBRegressor IMPORTANCES
xgb_fs = XGBRegressor(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0
)
xgb_fs.fit(X, y)

importances = pd.Series(xgb_fs.feature_importances_, index=feature_names)
top_feats  = importances.nlargest(TOPK).index.tolist()

print(f"Top {TOPK} features by XGB importance:\n", importances.nlargest(TOPK), "\n")

# 4) REDUCE TO TOP-K
X_top = df[top_feats].values

# 5) 10-FOLD CV WITH XGBRegressor
xgb_reg = XGBRegressor(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0
)
cv_scores_r2 = cross_val_score(
    xgb_reg, X_top, y,
    cv=10,
    scoring="r2",
    n_jobs=-1
)

print(f"CV R²: mean = {cv_scores_r2.mean():.3f}  ±  {cv_scores_r2.std():.3f}")
