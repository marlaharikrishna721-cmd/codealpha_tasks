# 🚗 Beautiful Car Price Prediction App

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Page Config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 30px;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    .result {
        font-size: 28px;
        font-weight: bold;
        color: #27ae60;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# Title
st.markdown('<div class="title">🚗 Car Price Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict the resale value of your car using Machine Learning</div>', unsafe_allow_html=True)


# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("car_data.csv")

df = load_data()

# Preprocessing
df = df.drop(['Car_Name'], axis=1)
df['Car_Age'] = 2025 - df['Year']
df.drop(['Year'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


# Layout: Two Columns
col1, col2 = st.columns([1, 1])

# LEFT SIDE → INPUTS
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔧 Enter Car Details")

    present_price = st.number_input("💰 Present Price (Lakhs)", min_value=0.0)
    kms_driven = st.number_input("📍 Kilometers Driven", min_value=0)
    owner = st.selectbox("👤 Number of Owners", [0, 1, 2, 3])
    car_age = st.slider("📅 Car Age (Years)", 0, 20)

    fuel_type = st.selectbox("⛽ Fuel Type", ["Petrol", "Diesel"])
    seller_type = st.selectbox("🏪 Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("⚙️ Transmission", ["Manual", "Automatic"])

    predict_btn = st.button("🚀 Predict Price")

    st.markdown('</div>', unsafe_allow_html=True)


# RIGHT SIDE → RESULT
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if predict_btn:
        input_dict = {
            'Present_Price': present_price,
            'Kms_Driven': kms_driven,
            'Owner': owner,
            'Car_Age': car_age,
            'Fuel_Type_Diesel': 1 if fuel_type == "Diesel" else 0,
            'Fuel_Type_Petrol': 1 if fuel_type == "Petrol" else 0,
            'Seller_Type_Individual': 1 if seller_type == "Individual" else 0,
            'Transmission_Manual': 1 if transmission == "Manual" else 0
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=X.columns, fill_value=0)

        prediction = model.predict(input_df)

        st.markdown(
            f'<div class="result">💸 ₹ {round(prediction[0], 2)} Lakhs</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("Enter details and click Predict 🚀")

    st.markdown('</div>', unsafe_allow_html=True)


# Footer
st.markdown("""
<hr>
<center>Made with ❤️ using Streamlit</center>
""", unsafe_allow_html=True)