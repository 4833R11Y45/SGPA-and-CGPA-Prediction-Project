import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained models and preprocessing objects
rf_model = joblib.load('rf_sgpa5_model.pkl')  # Random Forest for SGPA5
ridge_model = joblib.load('ridge_cgpa5_model.pkl')  # Ridge Regression for CGPA5
scaler = joblib.load('min_max_scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Streamlit app layout
st.title("SGPA and CGPA Prediction")

# Collecting user inputs for the features
inter_perc = st.number_input("Enter Intermediate Percentage", min_value=0.0, max_value=100.0, step=0.01)
sgpa1 = st.number_input("Enter SGPA1", min_value=0.0, max_value=4.0, step=0.01)
sgpa2 = st.number_input("Enter SGPA2", min_value=0.0, max_value=4.0, step=0.01)
sgpa3 = st.number_input("Enter SGPA3", min_value=0.0, max_value=4.0, step=0.01)
sgpa4 = st.number_input("Enter SGPA4", min_value=0.0, max_value=4.0, step=0.01)
cs_interest_options = ['Agree', 'Disagree', 'Neutral', 'Strongly Agree', 'Strongly Disagree']
cs_interest = st.selectbox("Select CS Interest Level", options=cs_interest_options)

# Handling 'StudyMode' input
study_mode_options = ['Group study', 'Independent study']
study_mode_selected = st.selectbox("Select Study Mode", options=study_mode_options)

def process_user_input(cs_interest, inter_perc, sgpa1, sgpa2, sgpa3, sgpa4, study_mode):
    # Encode 'CS_Interest' using the label encoder
    cs_interest_encoded = label_encoders['CS_Interest'].transform([cs_interest])

    # Prepare the numerical features
    numerical_features = np.array([[inter_perc, sgpa1, sgpa2, sgpa3, sgpa4]])

    # Apply scaler to numerical features
    numerical_features_scaled = scaler.transform(numerical_features)

    # Handling 'StudyMode' one-hot encoding
    study_mode_encoded = [1 if study_mode == 'Group study' else 0, 
                          1 if study_mode == 'Independent study' else 0]

    # Combine all features into a single array
    features = np.concatenate([
        cs_interest_encoded.reshape(1, -1), 
        numerical_features_scaled, 
        np.array([study_mode_encoded])
    ], axis=1)

    return features

# Define a function to determine the performance comment based on GPA
def get_performance_comment(gpa):
    if 3.51 <= gpa <= 4.00:
        return "Extraordinary Performance"
    elif 3.00 <= gpa <= 3.50:
        return "Very Good Performance"
    elif 2.51 <= gpa <= 2.99:
        return "Good Performance"
    elif 2.00 <= gpa <= 2.50:
        return "Satisfactory Performance"
    elif 1.00 <= gpa <= 1.99:
        return "Poor Performance"
    elif 0.00 <= gpa <= 0.99:
        return "Very Poor Performance"
    else:
        return "Invalid GPA"

# Update the prediction part of your app.py with the following:
# Predict button
if st.button("Predict"):
    # Preprocess inputs
    processed_features = process_user_input(cs_interest, inter_perc, sgpa1, sgpa2, sgpa3, sgpa4, study_mode_selected)

    # Make predictions
    sgpa_pred = rf_model.predict(processed_features)[0]
    cgpa_pred = ridge_model.predict(processed_features)[0]

    # Get performance comments
    sgpa_comment = get_performance_comment(sgpa_pred)

    # Display predictions with comments
    st.write(f"Predicted SGPA: {sgpa_pred:.2f}")
    st.write(f"Predicted CGPA: {cgpa_pred:.2f}")
    st.write(f"Comment - {sgpa_comment}")
