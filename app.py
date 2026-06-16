import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Visit With Us Prediction")

# Example input form
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=1000, max_value=100000, value=50000)
family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3)

if st.button("Predict"):
    # Build input DataFrame
    X_input = pd.DataFrame([[age, income, family_size]], columns=["Age", "AnnualIncome", "FamilySize"])
    # Scale input
    X_input_scaled = scaler.transform(X_input)
    # Predict
    prediction = model.predict(X_input_scaled)
    st.write("Prediction:", "Product Taken" if prediction[0] == 1 else "Not Taken")
