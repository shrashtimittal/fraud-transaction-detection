import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as calculate_auc
from sklearn.metrics import precision_recall_curve, average_precision_score

# -----------------------------
# Load train/val splits
# -----------------------------
X_train = pd.read_pickle("data/X_train.pkl")
y_train = pd.read_pickle("data/y_train.pkl")
X_val = pd.read_pickle("data/X_val.pkl")
y_val = pd.read_pickle("data/y_val.pkl")

print("Train shape:", X_train.shape, "Val shape:", X_val.shape)
print("NaNs in train:", X_train.isnull().sum().sum())
print("NaNs in val:", X_val.isnull().sum().sum())

# -----------------------------
# Define models with Pipelines
# -----------------------------
models = {
    "LogisticRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=500, solver="lbfgs", n_jobs=-1))
    ]),
    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))
    ]),
    "XGBoost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
            random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric="logloss"
        ))
    ])
}

results = {}

# -----------------------------
# Train & Evaluate
# -----------------------------
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    print(f"\n{name} Classification Report:")
    print(classification_report(y_val, y_pred, digits=4))

    auc = roc_auc_score(y_val, y_prob)
    results[name] = auc
    print(f"{name} ROC-AUC: {auc:.4f}")

    # Save model
    joblib.dump(model, f"models/{name}.joblib")

# -----------------------------
# Plot ROC Curves
# -----------------------------
plt.figure(figsize=(8, 6))

for name, model in models.items():
    y_prob = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc_value = calculate_auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_value:.3f})")

plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison (Validation Set)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot Precision-Recall Curves
# -----------------------------
plt.figure(figsize=(8, 6))

for name, model in models.items():
    y_prob = model.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_prob)
    ap = average_precision_score(y_val, y_prob)
    plt.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison (Validation Set)")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Summary of results
# -----------------------------
print("\n=== Validation ROC-AUC Scores ===")
for name, auc in results.items():
    print(f"{name}: {auc:.4f}")
