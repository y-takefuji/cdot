import pandas as pd
from sklearn.ensemble        import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# PARAMETERS
BINS = 16
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 1) LOAD & DROP UNUSED COLUMNS
df = pd.read_csv("cdot_data.csv")
df.drop(columns=[
    "project_number",
    "engineers_estimate",
    "bid_days",
    "start_date"
], inplace=True)

# 2) EXTRACT FEATURES & BIN TARGET INTO 'BINS' CLASSES
y = pd.qcut(df.pop("bid_total"), q=BINS, labels=False).astype(int)
X = df.values
feature_names = df.columns

# 3) RANK FEATURES WITH ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
et.fit(X, y)
importances = pd.Series(et.feature_importances_, index=feature_names)
top_feats = importances.nlargest(10).index.tolist()

print("Top 10 features by importance:\n", importances.nlargest(10))

# 4) REDUCE TO TOP-10 & SPLIT
X_top = df[top_feats].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X_top, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# 5) 10-FOLD CV ON TRAINING SET
et_final = ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
cv_scores = cross_val_score(
    et_final, X_tr, y_tr,
    cv=10, scoring="accuracy", n_jobs=-1
)

print(f"\nCV mean accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
