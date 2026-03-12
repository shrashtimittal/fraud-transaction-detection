# 💳 Fraud Transaction Detection

## 📌 Project Overview
This project implements a machine learning pipeline for detecting fraudulent financial transactions. The system analyzes transaction attributes such as transaction amount, customer behavior, and terminal activity to classify whether a transaction is legitimate or fraudulent.

Fraud detection is a critical application of machine learning in financial systems, helping institutions detect suspicious activities and prevent financial losses.

---

## 🎯 Objective
The goal of this project is to build a machine learning model capable of accurately identifying fraudulent transactions using transaction data.

The system performs:
- Data preprocessing
- Exploratory data analysis
- Feature engineering
- Model training and tuning
- Fraud prediction

---

## 🗂 Dataset
The dataset used in this project is a **simulated financial transaction dataset** containing both legitimate and fraudulent transactions.

Fraud scenarios included in the dataset:

1. Transactions above a certain threshold amount are marked as fraudulent.
2. Fraudulent activity originating from compromised payment terminals.
3. Customer accounts performing abnormal spending behavior due to leaked credentials.

Key dataset features:

| Column | Description |
|------|-------------|
| TRANSACTION_ID | Unique transaction identifier |
| TX_DATETIME | Date and time of transaction |
| CUSTOMER_ID | Unique customer identifier |
| TERMINAL_ID | Merchant terminal identifier |
| TX_AMOUNT | Transaction amount |
| TX_FRAUD | Fraud label (0 = Legitimate, 1 = Fraud) |

Due to size limitations, the dataset is **not included in this repository**.

---

## 🧠 Machine Learning Models Used

Several models were trained and evaluated:

- Logistic Regression
- Random Forest
- XGBoost

The best performing models were saved for deployment.

---

## 📦 Pretrained Models

### Included in Repository
models/best_xgboost.pkl

### Large Model Download (Random Forest)

Due to GitHub's file size limit (100MB), the Random Forest model is hosted externally.

Download here:

https://drive.google.com/file/d/1_HhKgvmPDDnjKwT2R4RiLN_ZeHjSxnhF/view?usp=sharing

After downloading:

1. Extract the ZIP file
2. Place the model file inside:
models/

Example:
models/
├── best_xgboost.pkl
└── best_randomforest.pkl

---

## ⚙️ Installation

Clone the repository
git clone https://github.com/shrashtimittal/fraud-transaction-detection.git

cd fraud-transaction-detection

Install dependencies
pip install -r requirements.txt

---

## 🚀 Running the Project

You can run training or evaluation scripts from the **src** directory.

Example:
python src/train_models.py

For predictions:
python src/predict.py

---

## 📂 Project Structure
fraud-transaction-detection
│
├── models
│ └── best_xgboost.pkl
│
├── src
│ ├── analyze_results.py
│ ├── app.py
│ ├── data_loader.py
│ ├── eda_plots.py
│ ├── eda_split.py
│ ├── evaluate.py
│ ├── predict.py
│ ├── save_best_models.py
│ ├── train_models.py
│ └── tune_models.py
│
├── requirements.txt
└── README.md

---

## 📊 Key Insights

- Fraudulent transactions often show abnormal transaction amounts.
- Terminal based fraud patterns can be detected through transaction clustering.
- Customer behavior analysis improves fraud detection accuracy.

---

## 🔮 Future Improvements

- Real-time fraud detection pipeline
- Integration with streaming transaction data
- Deployment as an API service
- Advanced anomaly detection models

---

## 👩‍💻 Author

Shrashti Mittal  
Machine Learning & AI Enthusiast
