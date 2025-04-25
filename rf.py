import pandas as pd
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

# PARAMETERS
BINS         = 16
TEST_SIZE    = 0.20
RANDOM_STATE = 42
TOPK         = 50

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
X = df.values
feature_names = df.columns

# 3) RANK FEATURES USING RandomForestClassifier
rf_fs = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_fs.fit(X, y)

importances = pd.Series(rf_fs.feature_importances_, index=feature_names)
top_feats  = importances.nlargest(TOPK).index.tolist()

print("Top 10 features by importance:\n", importances.nlargest(TOPK))

# 4) REDUCE TO TOP-10 AND SPLIT
X_top = df[top_feats].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X_top, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# 5) 10-FOLD CV ON TRAINING SET
rf_clf = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
cv_scores = cross_val_score(
    rf_clf, X_tr, y_tr,
    cv=10,
    scoring="accuracy",
    n_jobs=-1
)

print(f"\nCV mean accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
