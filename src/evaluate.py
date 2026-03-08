# src/evaluate.py
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# -----------------------------
# Load test data
# -----------------------------
X_test = pd.read_pickle("data/X_test.pkl")
y_test = pd.read_pickle("data/y_test.pkl")

print("Test shape:", X_test.shape, "Labels:", y_test.shape)

# -----------------------------
# Load tuned models
# -----------------------------
rf_model = joblib.load("models/best_randomforest.pkl")
xgb_model = joblib.load("models/best_xgboost.pkl")

# -----------------------------
# Evaluate each model
# -----------------------------
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

report_path = os.path.join(results_dir, "test_evaluation.txt")
with open(report_path, "w") as f:

    for name, model in [("RandomForest", rf_model), ("XGBoost", xgb_model)]:
        print(f"\nEvaluating {name}...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Classification report
        report = classification_report(y_test, y_pred, digits=4)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Save to file
        f.write(f"\n{name} Classification Report:\n{report}\n")
        f.write(f"{name} ROC-AUC: {roc_auc:.4f}\n")
        f.write("=" * 60 + "\n")

        print(f"{name} ROC-AUC: {roc_auc:.4f}")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

# -----------------------------
# Final ROC curve plot
# -----------------------------
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on Test Set")
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, "roc_curve_test.png"))
plt.show()

print(f"\n✅ Results saved to {report_path} and ROC curve saved to results/roc_curve_test.png")
