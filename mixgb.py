import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy  as np
import pandas as pd

from xgboost                  import XGBRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection   import cross_val_score, KFold

# -------------------------------------------------------------------
# 1) LOAD & CLEAN
# -------------------------------------------------------------------
df = pd.read_csv("cdot_data.csv").fillna(0)
df = df.drop(columns=[
    "project_number",
    "engineers_estimate",
    "bid_days",
    "start_date"
], errors="ignore")

# -------------------------------------------------------------------
# 2) EXTRACT CONTINUOUS TARGET
# -------------------------------------------------------------------
y = df.pop("bid_total").values

# -------------------------------------------------------------------
# 3) RANK FEATURES BY MUTUAL INFORMATION → y
# -------------------------------------------------------------------
X = df.copy()
mi_scores = mutual_info_regression(
    X, y,
    discrete_features='auto',
    random_state=0
)
mi_series = pd.Series(mi_scores, index=X.columns) \
              .sort_values(ascending=False)

top_feats = mi_series.iloc[:100].index.tolist()  # Changed from 50 to 100

print("\nTop 100 features by mutual information:")  # Updated text
print(mi_series.iloc[:100])  # Changed from 50 to 100

# -------------------------------------------------------------------
# 4) SELECT TOP-100 & RUN XGBRegressor WITH 10-FOLD CV  # Updated comment
# -------------------------------------------------------------------
X_sel = X[top_feats].values

xgb = XGBRegressor(
    n_estimators=493,
    random_state=0,
    n_jobs=-1,
    learning_rate=0.01,
    verbosity=0
)
kf = KFold(n_splits=10, shuffle=True, random_state=54)

# scoring="r2" for R², or "neg_mean_squared_error" for MSE
scores = cross_val_score(
    xgb,
    X_sel,
    y,
    cv=kf,
    scoring="r2",
    n_jobs=-1
)

print(f"\n10-fold CV R² mean ± SD: {scores.mean():.3f} ± {scores.std():.3f}")
