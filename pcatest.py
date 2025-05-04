import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# 1) LOAD & DROP UNUSED
df = pd.read_csv("cdot_data.csv")
df.drop(columns=[
    "project_number",
    "engineers_estimate",
    "bid_days",
    "start_date"
], inplace=True)

print(f"Full dataset: {df.shape[0]} samples, {df.shape[1]} features")

# 2) EXTRACT TARGET (not used by PCA)
y = df.pop("bid_total")
print(f"Features only: {df.shape[0]} samples, {df.shape[1]} features\n")

# Helper to compute PCA‐based feature importances
def pca_feature_importance(X, n_components=None):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    # absolute loadings
    abs_loads = np.abs(pca.components_)           # shape = (n_pc, n_features)
    # weight by explained variance ratio
    weights   = pca.explained_variance_ratio_[:, None]
    weighted  = abs_loads * weights               # same shape
    # sum over PCs → one score per feature
    scores    = weighted.sum(axis=0)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)

# 3) SET 1: PCA importances on ALL features
feat_imp_full = pca_feature_importance(df)
set1 = feat_imp_full.head(10)

print("Set 1 – Top 10 features by PCA‐based importance (all features):")
print(set1, "\n")

# 4) SAVE top 100 features
top100 = feat_imp_full.head(100)
top100.to_frame(name="importance").to_csv("100.csv")

# 5) SET 2: PCA importances on the top-100 feature subset
df100 = df[top100.index]
feat_imp_100 = pca_feature_importance(df100)
set2 = feat_imp_100.head(10)

print("Set 2 – Top 10 features by PCA‐based importance (within top 100):")
print(set2)
