import pandas as pd
import numpy as np
from sklearn.decomposition   import PCA
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# PARAMETERS
BINS         = 16
TEST_SIZE    = 0.20
RANDOM_STATE = 42
TOPK         = 10  # number of principal components & number of features to report

# 1) LOAD & DROP UNUSED COLUMNS
df = pd.read_csv("cdot_data.csv")
df.drop(columns=[
    "project_number",
    "engineers_estimate",
    "bid_days",
    "start_date"
], inplace=True)

# 2) EXTRACT TARGET & BIN INTO 16 CLASSES
y = pd.qcut(df.pop("bid_total"), q=BINS, labels=False).astype(int)

# 3) PCA → TOPK COMPONENTS
X = df.values
pca = PCA(n_components=TOPK, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)

print(f"Explained variance by first {TOPK} components: "
      f"{pca.explained_variance_ratio_.sum():.3f}")

# 4) IDENTIFY TOP ORIGINAL FEATURES BY LOADING MAGNITUDE
#    Sum absolute loadings across components for each original feature
abs_loadings = np.abs(pca.components_).sum(axis=0)
feat_imp = pd.Series(abs_loadings, index=df.columns)
top_feats = feat_imp.nlargest(TOPK).index.tolist()

print("\nTop 10 original features by summed |loading|:")
for i, f in enumerate(top_feats, 1):
    print(f"{i:2d}. {f} (score={feat_imp[f]:.3f})")

# 5) REDUCE TO THOSE TOP-10 FEATURES & SPLIT
X_top = df[top_feats].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X_top, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# 6) 10‐FOLD CV WITH RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
cv_scores = cross_val_score(
    rf, X_tr, y_tr,
    cv=10,
    scoring="accuracy",
    n_jobs=-1
)

print(f"\nCV mean accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
