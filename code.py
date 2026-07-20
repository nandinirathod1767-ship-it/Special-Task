import streamlit as st
import pandas as pd
import joblib

# Load Files
model = joblib.load("insurance_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("Insurance Charges Prediction")

# User Inputs
age = st.number_input("Age", 18, 100, 25)

sex = st.selectbox("Gender", ["male", "female"])

bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

children = st.number_input("Children", 0, 10, 0)

smoker = st.selectbox("Smoker", ["yes", "no"])

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# Create DataFrame
input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "bmi": [bmi],
    "children": [children],
    "smoker": [smoker],
    "region": [region]
})

# One-Hot Encoding
input_encoded = pd.get_dummies(input_data)

# Add Missing Columns
for col in columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Arrange Columns
input_encoded = input_encoded[columns]

# Scale Numerical Columns
numerical_columns = ["age", "bmi", "children"]
input_encoded[numerical_columns] = scaler.transform(
    input_encoded[numerical_columns]
)

# Prediction
if st.button("Predict Charges"):
    prediction = model.predict(input_encoded)
    st.success(f"Predicted Insurance Charges = ₹ {prediction[0]:,.2f}")