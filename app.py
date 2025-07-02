import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from huggingface_hub import hf_hub_download

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="🛡️ Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Custom Styling ----------------
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: #222831;
        margin-bottom: 10px;
    }
    .subheader {
        text-align: center;
        font-size: 1.2rem;
        color: #393E46;
        margin-bottom: 30px;
    }
    .section {
        background-color: #f1f5f9;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .prediction {
        padding: 1rem;
        margin: 1rem 0;
        # background-color: #e0f7fa;
        border-left: 8px solid #00796b;
        border-radius: 6px;
    }
    .fraud {
        border-left-color: #c62828;
        # background-color: #ffebee;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Load Artifacts from Hugging Face ----------------
@st.cache_resource
def load_artifacts():
    repo_id = "PavanKumarD/Credit_Card_Fraud_Models"

    scaler = joblib.load(hf_hub_download(repo_id=repo_id, filename="scaler.pkl"))
    encoders = joblib.load(hf_hub_download(repo_id=repo_id, filename="label_encoders.pkl"))
    models = {
        "Logistic Regression": joblib.load(hf_hub_download(repo_id=repo_id, filename="Logistic Regression_model.pkl")),
        "Decision Tree": joblib.load(hf_hub_download(repo_id=repo_id, filename="Decision Tree_model.pkl")),
        "Random Forest": joblib.load(hf_hub_download(repo_id=repo_id, filename="Random Forest_model.pkl")),
        "XGBoost": joblib.load(hf_hub_download(repo_id=repo_id, filename="xgboost_model.pkl")),
    }
    return scaler, encoders, models

# ✅ Call the function to load everything
scaler, encoders, models = load_artifacts()

feature_order = [
    'cc_num', 'merchant', 'category', 'amt', 'gender', 'city', 'state', 'zip', 'lat', 'long',
    'city_pop', 'job', 'unix_time', 'merch_lat', 'merch_long', 'trans_hour', 'trans_day_of_week'
]

categorical_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']

# ---------------- Header ----------------
st.markdown('<div class="title">🛡️ Credit Card Fraud Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Predict whether a transaction is fraudulent using pre-trained ML models</div>', unsafe_allow_html=True)

# ---------------- Input Form ----------------
with st.form("fraud_form"):
    st.markdown("### 📝 Transaction Details")
    col1, col2 = st.columns(2)

    with col1:
        cc_num = st.number_input("💳 Credit Card Number", value=123456789.0)
        amt = st.number_input("💰 Transaction Amount", value=100.0)
        zip_code = st.number_input("📮 ZIP Code", value=10001.0)
        lat = st.number_input("🌐 Latitude", value=40.0)
        long = st.number_input("🌐 Longitude", value=-75.0)
        city_pop = st.number_input("🏙️ City Population", value=8500000.0)

    with col2:
        unix_time = st.number_input("🕒 Unix Timestamp", value=float(datetime.now().timestamp()))
        merch_lat = st.number_input("🏪 Merchant Latitude", value=40.0)
        merch_long = st.number_input("🏪 Merchant Longitude", value=-75.0)
        trans_hour = st.slider("⏰ Transaction Hour", 0, 23, 12)
        trans_day = st.slider("📅 Day of Week", 0, 6, 3)

    st.markdown("### 🔤 Categorical Details")
    merchant = st.text_input("Merchant", value="fraud_Rippin, Kub and Mann")
    category = st.selectbox("Category", encoders['category'].classes_.tolist())
    gender = st.selectbox("Gender", encoders['gender'].classes_.tolist())
    city = st.text_input("City", value="New York")
    state = st.selectbox("State", encoders['state'].classes_.tolist())
    job = st.selectbox("Job", encoders['job'].classes_.tolist())

    submit = st.form_submit_button("🔍 Predict")

# ---------------- Prediction Section ----------------
if submit:
    try:
        sample_input = {
            'cc_num': cc_num,
            'merchant': encoders['merchant'].transform([merchant])[0] if merchant in encoders['merchant'].classes_ else 0,
            'category': encoders['category'].transform([category])[0],
            'amt': amt,
            'gender': encoders['gender'].transform([gender])[0],
            'city': encoders['city'].transform([city])[0] if city in encoders['city'].classes_ else 0,
            'state': encoders['state'].transform([state])[0],
            'zip': zip_code,
            'lat': lat,
            'long': long,
            'city_pop': city_pop,
            'job': encoders['job'].transform([job])[0],
            'unix_time': unix_time,
            'merch_lat': merch_lat,
            'merch_long': merch_long,
            'trans_hour': trans_hour,
            'trans_day_of_week': trans_day
        }

        input_df = pd.DataFrame([sample_input], columns=feature_order)
        input_scaled = scaler.transform(input_df)

        st.markdown("### 📊 Prediction Results")
        for name, model in models.items():
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]
            fraud = pred == 1

            css_class = "fraud prediction" if fraud else "prediction"
            label = "🚨 Fraudulent" if fraud else "✅ Legitimate"

            st.markdown(f"""
                <div class="{css_class}">
                    <strong>{name}</strong><br>
                    {label} — <b>{prob:.2%}</b> probability of fraud
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
