# -*- coding: utf-8 -*-
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np
import pandas as pd

from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.ensemble   import RandomForestRegressor, ExtraTreesRegressor
from mlxtend.regressor  import StackingRegressor
import lightgbm as lgb
import xgboost as xgb

# 1) IMPORT TE + DEFINE SURROGATE‐BASED ETE
from pyinform.transferentropy import transfer_entropy

def compute_surrogate_TE(source, target, k=1, n_surrogates=100):
    surrogate_tes = []
    for _ in range(n_surrogates):
        shuffled_source = np.random.permutation(source)
        try:
            te = transfer_entropy(shuffled_source, target, k=k)
            surrogate_tes.append(te)
        except Exception:
            continue
    return np.mean(surrogate_tes) if surrogate_tes else 0.0

def compute_ETE(source, target, k=1, delay=1, n_surrogates=100):
    if len(source) <= delay or len(target) <= delay:
        return 0.0
    s = source[:-delay].astype(int)
    t = target[ delay:].astype(int)
    if len(np.unique(s)) < 2 or len(np.unique(t)) < 2:
        return 0.0
    try:
        te_orig      = transfer_entropy(s, t, k=k)
        te_surr_mean = compute_surrogate_TE(s, t, k=k, n_surrogates=n_surrogates)
        return max(0.0, te_orig - te_surr_mean)
    except Exception:
        return 0.0

# 2) LOAD & CLEAN
filename = 'cdot_data.csv'
print("loading:", filename)
df = pd.read_csv(filename).fillna(0)
df = df.drop(columns=[
    "project_number",
    "engineers_estimate",
    "bid_days",
    "start_date"
])

# 3) TARGET
y_cont = df.pop("bid_total").values
y_digitized, bins = pd.qcut(
    y_cont, 16, retbins=True,
    labels=False, duplicates='drop'
)
y = y_digitized.astype(int)

# 4) FEATURES and discretization
X = df.copy()
n_samples, n_features = X.shape
print("n_samples, n_features:", n_samples, n_features)

# ← Moved out of comment so it actually runs
X_disc = X.apply(
    lambda col: pd.cut(col,
                       bins=16,
                       labels=False,
                       duplicates='drop')
).astype(int)

# 5) COMPUTE ETE FEATURE → TARGET
ete_scores = {}
k_history  = 1
n_surr      = 100   # change if you like fewer/more surrogates
delay       = 1

for feat in X_disc.columns:
    x_series = X_disc[feat].values
    ete_val  = compute_ETE(
        x_series, y,
        k=k_history,
        delay=delay,
        n_surrogates=n_surr
    )
    ete_scores[feat] = ete_val

# pick top 50 by ETE
ete_series = pd.Series(ete_scores).sort_values(ascending=False)
top_n      = 50
top_feats  = ete_series.iloc[:top_n].index.tolist()

print(f"Top {top_n} features by Effective Transfer Entropy:")
print(ete_series.iloc[:top_n])

# 6) CORRELATION‐BASED DEDUPLICATION on the _continuous_ X
X_top50 = X[top_feats].copy()
corr    = X_top50.corr().abs()

thresh   = 0.95
selected = []
for feat in top_feats:
    if any(corr.loc[feat, sel] > thresh for sel in selected):
        continue
    selected.append(feat)

print(f"\nAfter dropping inter‐feature corr > {thresh}, kept {len(selected)} features:")
print(selected)

X_reduced = X_top50[selected]

# 7) STACKING & CV (unchanged)
trees   = 493
crossv  = 10
from sklearn.model_selection import train_test_split
cv      = ShuffleSplit(n_splits=crossv, test_size=0.2, random_state=54)

lg  = lgb.LGBMRegressor(num_leaves=31,
                        learning_rate=0.48,
                        n_estimators=182,
                        force_col_wise=True,
                        n_jobs=-1)
xg  = xgb.XGBRegressor(n_estimators=trees,
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

scores = cross_val_score(sclf, X_reduced, y, cv=cv)
print(scores, scores.mean(), round(scores.std(),4))
