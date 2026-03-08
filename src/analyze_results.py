# src/analyze_results.py
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------------------------------------
# Load test data
# -------------------------------------------------------
X_test = pd.read_pickle("data/X_test.pkl")
y_test = pd.read_pickle("data/y_test.pkl")
print("Test shape:", X_test.shape, "Labels:", y_test.shape)

# -------------------------------------------------------
# Load best models
# -------------------------------------------------------
rf_model = joblib.load("models/best_randomforest.pkl")
xgb_model = joblib.load("models/best_xgboost.pkl")

os.makedirs("results", exist_ok=True)

# -------------------------------------------------------
# Confusion Matrices
# -------------------------------------------------------
for name, model in [("RandomForest", rf_model), ("XGBoost", xgb_model)]:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(f"results/{name.lower()}_confusion_matrix.png")
    plt.close()
    print(f"✅ Saved {name} confusion matrix")

# -------------------------------------------------------
# Feature Importances
# -------------------------------------------------------
def plot_feature_importances(model, X, name):
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:10]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
    plt.title(f"Top 10 Feature Importances - {name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"results/{name.lower()}_feature_importances.png")
    plt.close()
    print(f"✅ Saved {name} feature importances")

plot_feature_importances(rf_model, X_test, "RandomForest")
plot_feature_importances(xgb_model, X_test, "XGBoost")

print("\n🎉 Analysis complete. Results saved in results/ folder")
