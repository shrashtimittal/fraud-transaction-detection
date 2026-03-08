# src/predict.py
import pandas as pd
import os
import joblib

INPUT_FOLDER = "E:/fraud_detection_project/data"
OUTPUT_FILE = "results/predictions.csv"

# **List of features the model was trained on (from your error message)**
MODEL_FEATURES = ['CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']

# -------------------------
# Load new PKL transactions
# -------------------------
dfs = []
for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".pkl"):
        df_part = pd.read_pickle(os.path.join(INPUT_FOLDER, file))
        dfs.append(df_part)

X_new = pd.concat(dfs, ignore_index=True)

# Drop unnecessary columns
# '0' is included here to remove the problematic column if it exists in the raw data
drop_cols = ["TX_FRAUD", "TX_DATETIME", "TRANSACTION_ID", "TX_FRAUD_SCENARIO", "0"]
# The errors='ignore' ensures the script doesn't crash if a column in drop_cols is already missing
X_new = X_new.drop(columns=[c for c in drop_cols if c in X_new.columns], errors='ignore')

# Encode categorical IDs
for col in ["CUSTOMER_ID", "TERMINAL_ID"]:
    if col in X_new.columns:
        X_new[col] = X_new[col].astype("category").cat.codes

# Fill NaNs
# Note: The FutureWarning about downcasting is from pandas, not critical to the error fix.
X_new = X_new.fillna(X_new.median())

# **CRITICAL FIX:** Select only the features the model expects, in the correct order.
# This prevents the 'feature_names mismatch' error by excluding the unwanted '0' column.
X_new = X_new[MODEL_FEATURES]

print("Loaded new data:", X_new.shape)

# -------------------------
# Load best model
# -------------------------
model = joblib.load("models/best_xgboost.pkl")  # or best_randomforest.pkl

# -------------------------
# Predict
# -------------------------
preds = model.predict(X_new)
probas = model.predict_proba(X_new)[:, 1]

results = pd.DataFrame({
    "prediction": preds,
    "fraud_probability": probas
})

results.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Predictions saved to {OUTPUT_FILE}")