import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy  as np
import pandas as pd

from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

# -------------------------------------------------------------------
# 0) IMPORT NPEET's k-NN ENTROPY ESTIMATOR
# -------------------------------------------------------------------
from npeet.entropy_estimators import entropy

def total_correlation(x, y, k=5):
    """
    Estimate TC(X;Y) = H(X) + H(Y) - H(X,Y) via k-NN.
    x, y are 1D numpy arrays of length n.
    """
    # make them column‐vectors
    x2 = x.reshape(-1,1).astype(float)
    y2 = y.reshape(-1,1).astype(float)
    Hx  = entropy(x2, k=k)
    Hy  = entropy(y2, k=k)
    Hxy = entropy(np.hstack([x2, y2]), k=k)
    return Hx + Hy - Hxy

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
# 2) EXTRACT TARGET & FEATURES
# -------------------------------------------------------------------
y = df.pop("bid_total").values.astype(float)
X = df.copy()
feat_names = X.columns.tolist()
n, p = X.shape
print(f"Loaded {n} samples × {p} features")

# -------------------------------------------------------------------
# 3) COMPUTE TOTAL CORRELATION(feature → target)
# -------------------------------------------------------------------
k_tc = 5
tc_scores = {}
print("\nEstimating TC(feature → bid_total)…")
for feat in feat_names:
    xi = X[feat].values
    try:
        tc_scores[feat] = total_correlation(xi, y, k=k_tc)
    except Exception:
        tc_scores[feat] = 0.0

# -------------------------------------------------------------------
# 4) RANK & SELECT TOP-100
# -------------------------------------------------------------------
tc_ser    = pd.Series(tc_scores).sort_values(ascending=False)
top_n     = min(100, len(tc_ser))  # Changed from 50 to 100
top_feats = tc_ser.index[:top_n].tolist()

print(f"\nTop {top_n} features by TC → bid_total:")
print(tc_ser.iloc[:top_n])

# -------------------------------------------------------------------
# 5) TRAIN RF WITH 10-FOLD CV
# -------------------------------------------------------------------
X_sel = X[top_feats].values
rf    = RandomForestRegressor(n_estimators=493,
                              random_state=0,
                              n_jobs=-1)
kf    = KFold(n_splits=10, shuffle=True, random_state=54)

scores = cross_val_score(rf, X_sel, y,
                         cv=kf,
                         scoring="r2",
                         n_jobs=-1)

print(f"\n10-fold CV R² mean ± SD: {scores.mean():.3f} ± {scores.std():.3f}")
