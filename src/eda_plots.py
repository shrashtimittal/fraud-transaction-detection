import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create results folder if not exists
os.makedirs("results", exist_ok=True)

# -----------------------------
# Load and combine dataset parts
# -----------------------------
dfs = []
for file in os.listdir("data"):
    if file.endswith(".pkl") and not file.startswith(("X_", "y_")):  # ignore split files
        df_part = pd.read_pickle(os.path.join("data", file))
        dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)

# Drop unwanted column if exists
if "0" in df.columns:
    df = df.drop(columns=["0"])

# Convert datetime column
df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])

print("Full dataset shape for plots:", df.shape)

# -----------------------------
# 1. Class imbalance plot
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="TX_FRAUD", data=df, palette="viridis")
plt.title("Fraud (1) vs Non-Fraud (0)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig("results/class_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# 2. Histogram of transaction amounts
# -----------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["TX_AMOUNT"], bins=100, kde=False, color="steelblue")
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.xlim(0, 2000)  # optional zoom to ignore extreme outliers
plt.savefig("results/tx_amount_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# 3. Time-series fraud frequency
# -----------------------------
daily_fraud = df.groupby(df["TX_DATETIME"].dt.date)["TX_FRAUD"].sum()
plt.figure(figsize=(12,5))
daily_fraud.plot(color="crimson")
plt.title("Daily Fraudulent Transactions Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Frauds")
plt.savefig("results/daily_fraud_trend.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ All plots saved in results/ folder")

