import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load model, scaler, encoders
model = joblib.load('trained_model.pkl')
scaler = joblib.load('scaler (1).pkl')
encoders = joblib.load('label_encoders.pkl')

# Define expected feature columns
features = ['Age', 'Gender', 'EducationBackground', 'MaritalStatus',
            'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency',
            'DistanceFromHome', 'EmpEducationLevel', 'EngagementIndex',
            'EmpEnvironmentSatisfaction', 'EmpHourlyRate', 'EmpJobInvolvement',
            'EmpJobLevel', 'EmpJobSatisfaction', 'NumCompaniesWorked',
            'OverTime', 'EmpLastSalaryHikePercent', 'OverallSatisfactionScore',
            'RoleTenureRatio', 'CompanyLoyalty', 'Attrition']  # ✅ Added Attrition

# Title
st.title("Employee Performance Prediction")

# Step 1: User input for essential fields
user_inputs = {
    'Age': st.slider('Age', 18, 60, 30),
    'Gender': st.selectbox('Gender', ['Male', 'Female']),
    'EmpDepartment': st.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources']),
    'EmpJobRole': st.selectbox('Job Role', ['Sales Executive', 'Research Scientist', 'Human Resources']),
    'OverTime': st.selectbox('OverTime', ['Yes', 'No']),
    'DistanceFromHome': st.slider('Distance from Home (km)', 0, 30, 5)
}

# Step 2: Default values for the rest
default_values = {
    'EducationBackground': 'Life Sciences',
    'MaritalStatus': 'Single',
    'BusinessTravelFrequency': 'Travel_Rarely',
    'EmpEducationLevel': 3,
    'EngagementIndex': 0.6,
    'EmpEnvironmentSatisfaction': 5,
    'EmpHourlyRate': 50,
    'EmpJobInvolvement': 3,
    'EmpJobLevel': 3,
    'EmpJobSatisfaction': 3,
    'NumCompaniesWorked': 5,
    'EmpLastSalaryHikePercent': 60,
    'OverallSatisfactionScore': 4,
    'RoleTenureRatio': 0.7,
    'CompanyLoyalty': 2,
    'Attrition': 'Yes'
}

# Step 3: Combine user and default input
full_input = {**default_values, **user_inputs}

# Step 4: Convert to DataFrame
input_df = pd.DataFrame([[full_input[feature] for feature in features]], columns=features)

# Step 5: Encode categorical features
for col in encoders:
    if col in input_df.columns:
        le = encoders[col]
        input_df[col] = le.transform(input_df[col])

# Step 6: Scale
input_scaled = scaler.transform(input_df)

# Step 7: Predict
if st.button("Predict Performance Rating"):
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Performance Rating: {prediction[0]}")


    

    
