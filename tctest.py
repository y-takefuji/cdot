import numpy as np
import pandas as pd
from npeet.entropy_estimators import entropy

def total_correlation(x, y, k=5):
    """
    Estimate TC(X;Y) = H(X) + H(Y) - H(X,Y) via k-NN.
    x, y are 1D numpy arrays of length n.
    """
    x2 = x.reshape(-1, 1).astype(float)
    y2 = y.reshape(-1, 1).astype(float)
    Hx  = entropy(x2, k=k)
    Hy  = entropy(y2, k=k)
    Hxy = entropy(np.hstack([x2, y2]), k=k)
    return Hx + Hy - Hxy

def tc_feature_importance(X, y, k=5):
    """
    Compute TC(feature, y) for each column in X.
    Returns a pandas.Series sorted descending by TC.
    """
    Xmat = np.asarray(X, dtype=float)
    yvec = np.asarray(y, dtype=float)
    scores = {}
    for i, colname in enumerate(X.columns):
        xi = Xmat[:, i]
        scores[colname] = total_correlation(xi, yvec, k=k)
    return pd.Series(scores).sort_values(ascending=False)

# ------------------------------------------------------------------
# 1) LOAD & DROP UNUSED
df = pd.read_csv("cdot_data.csv")
df.drop(columns=[
    "project_number",
    "engineers_estimate",
    "bid_days",
    "start_date"
], inplace=True)
print(f"Full dataset: {df.shape[0]} samples, {df.shape[1]} features")

# 2) EXTRACT TARGET 
y = df.pop("bid_total")
print(f"Features only: {df.shape[0]} samples, {df.shape[1]} features\n")

# 3) SET 1: Total-Correlation importances on ALL features
feat_imp_full_tc = tc_feature_importance(df, y, k=5)
set1_tc = feat_imp_full_tc.head(10)
print("Set 1 – Top 10 features by TC (all features):")
print(set1_tc, "\n")

# 4) SAVE top 100 features
top100_tc = feat_imp_full_tc.head(100)
top100_tc.to_frame(name="tc_score").to_csv("100_tc.csv")

# 5) SET 2: TC importances on the top-100 subset
df100 = df[top100_tc.index]
feat_imp_100_tc = tc_feature_importance(df100, y, k=5)
set2_tc = feat_imp_100_tc.head(10)
print("Set 2 – Top 10 features by TC (within top 100):")
print(set2_tc)
