import streamlit as st
import joblib
import pandas as pd

# Load the trained pipeline model
model = joblib.load('model_pipeline.pkl')

# Define function for prediction
def predict_heart_disease(features):
    # Convert the features into a DataFrame with the correct column names
    feature_names = ['General_Health', 'Exercise', 'Depression', 'Diabetes', 'Sex', 
                     'Age_Category', 'Smoking_History', 'Alcohol_Consumption', 
                     'Weight_(kg)', 'BMI']
    features_df = pd.DataFrame([features], columns=feature_names)
    
    # Make the prediction
    prediction = model.predict(features_df)
    return prediction[0]

# Streamlit UI elements
st.title("Heart Disease Prediction")

# Define the feature inputs
general_health = st.selectbox("General Health", options=["Poor", "Fair", "Good", "Very Good", "Excellent"])
exercise = st.selectbox("Exercise", options=["Yes", "No"])
depression = st.selectbox("Depression", options=["Yes", "No"])
diabetes = st.selectbox("Diabetes", options=["Yes", "No"])
sex = st.selectbox("Sex", options=["Male", "Female"])
age_category = st.selectbox("Age Category", options=[
    "1: 18-24", 
    "2: 25-29", 
    "3: 30-34", 
    "4: 35-39", 
    "5: 40-44", 
    "6: 45-49", 
    "7: 50-54", 
    "8: 55-59", 
    "9: 60-64", 
    "10: 65-69", 
    "11: 70-74", 
    "12: 75-79", 
    "13: 80+"
])
weight = st.number_input("Weight (kg)", min_value=0.0, value=70.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
smoking_history = st.selectbox("Smoking History", options=["Yes", "No"])
alcohol_consumption = st.selectbox("Alcohol Consumption", options=["Yes", "No"])

# Collect the feature inputs in the correct order
features = [
    general_health, exercise, depression, diabetes, sex, 
    age_category.split(":")[0].strip(),  # Only pass the numeric value
    smoking_history, alcohol_consumption, weight, bmi
]

# Prediction
if st.button("Predict"):
    result = predict_heart_disease(features)
    st.write(f"The model predicts that the patient {'has' if result == 1 else 'does not have'} heart disease.")
