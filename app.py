# Step 4: Model Deployment (Streamlit App)
!pip install streamlit
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Load model from Hugging Face Hub
model_path = hf_hub_download(repo_id="DrNiha555/visit_with_us-model", filename="model.pkl")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Purchase Predictor")

st.write("Fill in customer details to predict purchase likelihood.")

# Input fields (all attributes from data dictionary)
age = st.number_input("Age", min_value=18, max_value=100)
monthly_income = st.number_input("Monthly Income", min_value=0)
passport = st.selectbox("Passport", [0,1])
own_car = st.selectbox("Own Car", [0,1])
num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20)
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5)

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([[age, monthly_income, passport, own_car, num_trips, pitch_score]],
                            columns=["Age","MonthlyIncome","Passport","OwnCar","NumberOfTrips","PitchSatisfactionScore"])
    prediction = model.predict(input_df)[0]
    st.success("Prediction: " + ("Will Purchase" if prediction == 1 else "Will Not Purchase"))
