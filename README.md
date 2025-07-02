# 🛡️ Credit Card Fraud Detection App

A machine learning-powered web app that predicts whether a credit card transaction is fraudulent — using models loaded from Hugging Face 🤖 and built with Streamlit 💻.

---

## 🚀 Live App

🔗 **Try it now**: [Streamlit App](https://credit-card-fraud-detection-by-pavan.streamlit.app)

---

## 📦 Features

- 🧠 Pre-trained models: Logistic Regression, Decision Tree, Random Forest, XGBoost
- 📊 Real-time fraud predictions with probability scores
- 🌐 Clean, responsive UI with modern styling
- 🔄 Fully integrated with Hugging Face for model storage and loading

---

## 🧰 Tech Stack

| Tool            | Purpose                    |
|-----------------|----------------------------|
| `Streamlit`     | Frontend Web App           |
| `scikit-learn`  | ML Models & Encoding       |
| `xgboost`       | Gradient Boosting Model    |
| `joblib`        | Model Serialization        |
| `huggingface_hub` | Load models from Hugging Face |

---

## 🧪 Models Used

All models and preprocessing artifacts are hosted on [Hugging Face Hub](https://huggingface.co/PavanKumarD/Credit_Card_Fraud_Models).

- `scaler.pkl`
- `label_encoders.pkl`
- `Logistic Regression_model.pkl`
- `Decision Tree_model.pkl`
- `Random Forest_model.pkl`
- `xgboost_model.pkl`
## Project Structure
---
Credit-Card-Fraud-Detection/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
└── README.md              # Project overview

## 🧑‍💻 How to Run Locally

```bash
git clone https://github.com/Pavan-Kumar-Dirisala/Credit-Card-Fraud-Detection
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
streamlit run app.py
