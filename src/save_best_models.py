import joblib
import os
from tune_models import best_rf, best_xgb 

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save them
joblib.dump(best_rf, "models/best_randomforest.pkl")
joblib.dump(best_xgb, "models/best_xgboost.pkl")

print("✅ Best models saved in models/ folder")
