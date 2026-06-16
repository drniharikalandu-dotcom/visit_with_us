import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# Always pull latest artifacts from Hugging Face Hub
repo_id = "DrNiha555/visit_with_us_model"
model_path = hf_hub_download(repo_id=repo_id, filename="model.pkl", repo_type="model")
scaler_path = hf_hub_download(repo_id=repo_id, filename="scaler.pkl", repo_type="model")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title("Visit With Us Prediction")

# Example input form
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=1000, max_value=100000, value=50000)
family_size = st.number_input("Family Size", min_value=1, max_value=10, value=3)

if st.button("Predict"):
    X_input = pd.DataFrame(
        [[age, income, family_size]],
        columns=["Age", "AnnualIncome", "FamilySize"]
    )
    X_input_scaled = scaler.transform(X_input)
    prediction = model.predict(X_input_scaled)
    st.write("Prediction:", "✅ Product Taken" if prediction[0] == 1 else "❌ Not Taken")
