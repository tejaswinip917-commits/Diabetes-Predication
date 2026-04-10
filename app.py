import streamlit as st
import pandas as pd
import joblib

# 1. Load the model and scaler
# These files must be in the same GitHub folder!
model = joblib.load('diabetes_model.joblib')
scaler = joblib.load('scaler.joblib')

st.title("🩺 Diabetes Prediction App")

# 2. Input fields
preg = st.number_input("Pregnancies", 0, 20, 1)
gluco = st.number_input("Glucose", 0, 200, 100)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    # 3. Create feature list (Ensure order matches your training data!)
    features = [[preg, gluco, bp, skin, insulin, bmi, dpf, age]] 
    
    # 4. Scale and Predict
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    
    if prediction[0] == 1:
        st.error("The model predicts a high risk of diabetes.")
    else:
        st.success("The model predicts a low risk of diabetes.")