import pandas as pd
import numpy as np
from sklearn.decomposition    import PCA
from xgboost                  import XGBRegressor
from sklearn.model_selection  import cross_val_score
from sklearn.metrics          import r2_score, mean_squared_error

# PARAMETERS
TEST_SIZE    = 0.20
RANDOM_STATE = 54
TOPK         = 100  # number of principal components & features to keep

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

# 3) PCA → TOPK COMPONENTS
X = df.values
pca = PCA(n_components=TOPK, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)

print(f"Explained variance by first {TOPK} components: "
      f"{pca.explained_variance_ratio_.sum():.3f}")

# 4) IDENTIFY TOP ORIGINAL FEATURES BY LOADING MAGNITUDE
abs_loadings = np.abs(pca.components_).sum(axis=0)
feat_imp     = pd.Series(abs_loadings, index=df.columns)
top_feats    = feat_imp.nlargest(TOPK).index.tolist()

print("\nTop 50 original features by summed |loading|:")
for i, f in enumerate(top_feats, 1):
    print(f"{i:2d}. {f} (score={feat_imp[f]:.3f})")

# 5) REDUCE TO THOSE TOP-50 FEATURES
X_top = df[top_feats].values

# 6) 10‐FOLD CV WITH XGBRegressor
xgb = XGBRegressor(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0
)
cv_scores = cross_val_score(
    xgb, X_top, y,
    cv=10,
    scoring="r2",    # or "neg_mean_squared_error"
    n_jobs=-1
)

print(f"\nCV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
