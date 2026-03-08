# src/tune_models.py
import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------------------------------------
# Load train/val data
# -------------------------------------------------------
X_train = pd.read_pickle("data/X_train.pkl")
y_train = pd.read_pickle("data/y_train.pkl")
X_val = pd.read_pickle("data/X_val.pkl")
y_val = pd.read_pickle("data/y_val.pkl")

print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

# -------------------------------------------------------
# Define smaller parameter grids
# (≈10 candidates each → 30 fits per model)
# -------------------------------------------------------
rf_params = {
    "n_estimators": [200, 400],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

xgb_params = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
}

rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
xgb = XGBClassifier(
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric="logloss"
)

# -------------------------------------------------------
# Run tuning only if script is executed directly
# -------------------------------------------------------
if __name__ == "__main__":

    # ---------------- RandomForest ----------------
    print("\nTuning RandomForest...")
    rf_grid = GridSearchCV(rf, rf_params, scoring="roc_auc", cv=3, verbose=2, n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print("Best RF Params:", rf_grid.best_params_)

    y_pred = best_rf.predict(X_val)
    y_proba = best_rf.predict_proba(X_val)[:, 1]
    print("\nRandomForest tuned Classification Report:")
    print(classification_report(y_val, y_pred))
    print("ROC-AUC:", roc_auc_score(y_val, y_proba))

    # ---------------- XGBoost ----------------
    print("\nTuning XGBoost...")
    xgb_grid = GridSearchCV(xgb, xgb_params, scoring="roc_auc", cv=3, verbose=2, n_jobs=-1)
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    print("Best XGBoost Params:", xgb_grid.best_params_)

    y_pred = best_xgb.predict(X_val)
    y_proba = best_xgb.predict_proba(X_val)[:, 1]
    print("\nXGBoost tuned Classification Report:")
    print(classification_report(y_val, y_pred))
    print("ROC-AUC:", roc_auc_score(y_val, y_proba))

    # ---------------- Save models ----------------
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_rf, "models/best_randomforest.pkl")
    joblib.dump(best_xgb, "models/best_xgboost.pkl")

    print("\n✅ Best models saved in models/ folder")

# -------------------------------------------------------
# Export best models for importing elsewhere
# -------------------------------------------------------
try:
    best_rf
    best_xgb
except NameError:
    best_rf, best_xgb = None, None
