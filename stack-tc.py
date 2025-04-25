# -*- coding: utf-8 -*-
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np
import pandas as pd

from sklearn.model_selection        import ShuffleSplit, cross_val_score, train_test_split
from sklearn.ensemble              import RandomForestRegressor, ExtraTreesRegressor
from mlxtend.regressor             import StackingRegressor
import lightgbm as lgb
import xgboost as xgb

from sklearn.metrics               import mutual_info_score

# 1) LOAD & CLEAN
filename = 'cdot_data.csv'
print("loading:", filename)
df = pd.read_csv(filename).fillna(0)
df = df.drop(columns=[
    "project_number",
    "engineers_estimate",
    "bid_days",
    "start_date"
])

# 2) EXTRACT TARGET AND BIN/CAST TO INTEGER
y_cont = df.pop("bid_total").values  # keep continuous for saving later
y_digitized, bins = pd.qcut(
    y_cont, 16, retbins=True,
    labels=False, duplicates='drop'
)
y = y_digitized.astype(int)

# 3) FEATURES
X = df.copy()
n_samples, n_features = X.shape
print("n_samples, n_features:", n_samples, n_features)

# discretize each feature into 16 bins (needed for mutual_info_score)
X_disc = X.apply(
    lambda col: pd.cut(col, bins=16,
                       labels=False,
                       duplicates='drop')
).astype(int)

# 4) COMPUTE TOTAL CORRELATION (Mutual Information) FEATURE → TARGET
tc_scores = {}
for feat in X_disc.columns:
    tc = mutual_info_score(X_disc[feat].values, y)
    tc_scores[feat] = tc

# pick top 50 features by Total Correlation
tc_series = pd.Series(tc_scores).sort_values(ascending=False)
top_n = 50
top_feats = tc_series.iloc[:top_n].index.tolist()

print(f"Top {top_n} features by Total Correlation (MI):")
print(tc_series.iloc[:top_n])

# 5) CORRELATION‐BASED DEDUPLICATION
X_top50 = X[top_feats].copy()
corr = X_top50.corr().abs()

thresh = 0.98
selected = []
for feat in top_feats:
    if any(corr.loc[feat, sel] > thresh for sel in selected):
        continue
    selected.append(feat)

print(f"\nAfter dropping inter‐feature corr > {thresh}, kept {len(selected)} features:")
print(selected)

X_reduced = X_top50[selected]

# --- NEW: save the 50 selected features + continuous target ---
small = X_reduced.copy()
small['bid_total'] = y_cont
small.to_csv('small.csv', index=False)
print("Saved reduced dataset (50 features + bid_total) to small.csv")

# 6) DEFINE STACKING REGRESSOR & CV (unchanged)
trees   = 493
crossv  = 10
cv      = ShuffleSplit(n_splits=crossv, test_size=0.2, random_state=54)

lg = lgb.LGBMRegressor(num_leaves=31,
                       learning_rate=0.48,
                       n_estimators=182,
                       force_col_wise=True,
                       n_jobs=-1)
xg = xgb.XGBRegressor(n_estimators=trees,
                      use_label_encoder=False,
                      eval_metric='rmse',
                      n_jobs=-1)
ext = ExtraTreesRegressor(n_estimators=trees,
                          n_jobs=-1,
                          random_state=0)
rf = RandomForestRegressor(n_estimators=trees,
                           n_jobs=-1,
                           random_state=0)

sclf = StackingRegressor(regressors=[lg, ext, xg],
                         meta_regressor=rf)

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=54
)

scores = cross_val_score(sclf, X, y, cv=cv)
print(scores, scores.mean(), round(scores.std(), 4))
