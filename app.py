import streamlit as st
import pandas as pd
import joblib

# Load model, scaler and columns
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Life Expectancy Predictor", page_icon="🌍")

st.title("🌍 Country Life Expectancy Prediction")
st.write("Enter the details below:")

gdp = st.number_input("GDP", value=10000.0)
inflation = st.number_input("Inflation (%)", value=5.0)
birth_rate = st.number_input("Birth Rate", value=20.0)
death_rate = st.number_input("Death Rate", value=7.0)
internet_pct = st.number_input("Internet Users (%)", value=70.0)
population = st.number_input("Population", value=1000000.0)
health_expenditure = st.number_input("Health Expenditure (% GDP)", value=5.0)
hospital_beds = st.number_input("Hospital Beds", value=3.0)

if st.button("Predict Life Expectancy"):

    input_data = pd.DataFrame([[

        gdp,
        inflation,
        birth_rate,
        death_rate,
        internet_pct,
        population,
        health_expenditure,
        hospital_beds

    ]], columns=columns)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    st.success(f"Predicted Life Expectancy: {prediction[0]:.2f} years")