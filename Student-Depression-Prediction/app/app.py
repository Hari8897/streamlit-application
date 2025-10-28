import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
MODEL_PATH = 'model/finalized_model.sav'
SCALER_PATH = 'model/scaler.sav'

# Load the model and scaler
model = pickle.load(open(MODEL_PATH, 'rb'))
scaler = pickle.load(open(SCALER_PATH, 'rb'))

# Streamlit Dashboard
def main():
    st.title("Student Depression Prediction Dashboard")
    st.write("""
    This app predicts the likelihood of depression in students based on various factors like 
    gender, sleep duration, dietary habits, and more.
    """)

    # User inputs
    st.sidebar.header("User Input Features")
    
    def user_input_features():
        gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
        age = st.sidebar.slider('Age', 16, 30, 20)
        academic_pressure = st.sidebar.slider('Academic Pressure (0-10)', 0, 10, 5)
        study_satisfaction = st.sidebar.slider('Study Satisfaction (0-10)', 0, 10, 5)
        sleep_duration = st.sidebar.selectbox(
            'Sleep Duration', 
            ('More than 8 hours', '7-8 hours', '5-6 hours', 'Less than 5 hours')
        )
        dietary_habits = st.sidebar.selectbox('Dietary Habits', ('Unhealthy', 'Moderate', 'Healthy'))
        suicidal_thoughts = st.sidebar.selectbox('Have you ever had suicidal thoughts?', ('Yes', 'No'))
        study_hours = st.sidebar.slider('Study Hours per Day', 0, 16, 6)
        financial_stress = st.sidebar.slider('Financial Stress (0-10)', 0, 10, 5)
        family_history = st.sidebar.selectbox('Family History of Mental Illness', ('Yes', 'No'))
        
        data = {
            'Gender': 0 if gender == 'Male' else 1,
            'Age': age,
            'Academic Pressure': academic_pressure,
            'Study Satisfaction': study_satisfaction,
            'Sleep Duration': {
                'More than 8 hours': 0,
                '7-8 hours': 1,
                '5-6 hours': 2,
                'Less than 5 hours': 3
            }[sleep_duration],
            'Dietary Habits': {'Unhealthy': 0, 'Moderate': 1, 'Healthy': 2}[dietary_habits],
            'Have you ever had suicidal thoughts ?': 1 if suicidal_thoughts == 'Yes' else 0,
            'Study Hours': study_hours,
            'Financial Stress': financial_stress,
            'Family History of Mental Illness': 1 if family_history == 'Yes' else 0,
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    st.subheader("User Input Features")
    st.write(input_df)

    # Preprocess the input data
    input_data_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    st.subheader("Prediction")
    st.write("Depression Detected" if prediction[0] == 1 else "No Depression Detected")

    st.subheader("Prediction Probability")
    st.write(f"Probability of Depression: {prediction_proba[0][1]*100:.2f}%")
    st.write(f"Probability of No Depression: {prediction_proba[0][0]*100:.2f}%")

if __name__ == '__main__':
    main()
