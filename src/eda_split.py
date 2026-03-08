import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load and clean dataset
# -----------------------------
dfs = []
for file in os.listdir("data"):
    if file.endswith(".pkl"):
        df_part = pd.read_pickle(os.path.join("data", file))
        dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)

# Drop unwanted column if exists
if "0" in df.columns:
    df = df.drop(columns=["0"])

# Drop rows without label
df = df.dropna(subset=["TX_FRAUD"])

# Ensure all column names are strings
df.columns = df.columns.astype(str)

# Convert categorical columns to numeric (simple label encoding)
for col in ["CUSTOMER_ID", "TERMINAL_ID"]:
    if col in df.columns:
        df[col] = df[col].astype("category").cat.codes

print("Cleaned shape:", df.shape)
print("Nulls after cleaning:\n", df.isnull().sum())

# -----------------------------
# Basic EDA
# -----------------------------
print("\nClass distribution:\n", df['TX_FRAUD'].value_counts())
sns.countplot(x="TX_FRAUD", data=df)
plt.title("Fraud (1) vs Non-Fraud (0)")
plt.show()

# -----------------------------
# Split
# -----------------------------
drop_cols = ["TX_FRAUD", "TX_DATETIME", "TRANSACTION_ID", "TX_FRAUD_SCENARIO", "0"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df["TX_FRAUD"].astype(int)

X = X.fillna(X.median())

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print("\nTrain:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
print("Fraud ratio in train:", y_train.mean())
print("Fraud ratio in val:", y_val.mean())
print("Fraud ratio in test:", y_test.mean())

# Save splits
X_train.to_pickle("data/X_train.pkl")
X_val.to_pickle("data/X_val.pkl")
X_test.to_pickle("data/X_test.pkl")
y_train.to_pickle("data/y_train.pkl")
y_val.to_pickle("data/y_val.pkl")
y_test.to_pickle("data/y_test.pkl")
