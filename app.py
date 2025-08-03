# Gender -> 1-Female, 0-Male
# Churn -> 1-Yes, 0-No
# Scaler is exported as scaler.pkl
# Model is exported as model.pkl
# Order of X -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np

# Load model and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Page configuration
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")

# Custom CSS styling with updated dark bluish purple background
st.markdown("""
    <style>
    body {
        background-color: #4e148c;
    }
    .main {
        background-color: #4e148c;
        color: white;
        padding: 2rem;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1.5rem;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1 style='text-align:center;'>📊 Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

# Input Description
st.markdown("### 🔍 Please enter customer details:")
st.markdown("Provide the required information and click **Predict!** to check if the customer is likely to churn.")
st.markdown("---")

# Input Fields in Columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🎂 Age", min_value=10, max_value=100, value=30)
    tenure = st.number_input("📅 Tenure (months)", min_value=0, max_value=130, value=10)

with col2:
    monthlycharge = st.number_input("💵 Monthly Charge", min_value=30, max_value=150)
    gender = st.selectbox("🧑 Gender", ["Male", "Female"])

st.markdown("---")

# Predict Button
predictbutton = st.button("🚀 Predict!")

# Prediction Logic
if predictbutton:
    gender_val = 0 if gender == "Male" else 1
    X = [age, gender_val, tenure, monthlycharge]
    X1 = np.array(X)

    try:
        X_array = scaler.transform([X1])
        prediction = model.predict(X_array)[0]
        predicted = "Yes 🔴" if prediction == 1 else "No 🟢"

        st.success(f"🎯 **Prediction Result:** Customer will churn? **{predicted}**")
        st.balloons()
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
else:
    st.info("📥 Enter all values and press the Predict button.")
