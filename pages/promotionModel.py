import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model

# Load the trained model and preprocessed data
model_path = 'employee_promotion_model.h5'  # Replace with your saved model path
model = load_model('models/employee_promotion_model.h5')  # Replace this with `load_model(model_path)` if saved
scaler = joblib.load('models/scaler.pkl')  # Replace with your actual scaler instance
label_encoders = joblib.load('models/label_encoders.pkl')  # Replace with your actual label encoders dictionary

# App title and description
st.title("Employee Promotion Prediction")
st.write("Use this tool to predict the likelihood of an employee being promoted based on their profile and performance data.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

# Categorical inputs with dropdown menus
department = st.sidebar.selectbox(
    'Department', 
    ['Sales & Marketing', 'Operations', 'Technology', 'HR']
)
region = st.sidebar.selectbox(
    'Region', 
    ['Chennai', 'Mumbai', 'Indore', 'Kochi', 'Bangalore']
)
education = st.sidebar.selectbox(
    'Education Level', 
    ["Bachelor's", "Master's", 'High School']
)
gender = st.sidebar.selectbox(
    'Gender', 
    ['m', 'f']
)
recruitment_channel = st.sidebar.selectbox(
    'Recruitment Channel', 
    ['linkedin', 'sourcing', 'other']
)

def get_numerical_input(label, min_value, max_value, default_value):
    """Function to get numerical input from the user."""
    input_value = st.sidebar.text_input(
        f"{label} (Range: {min_value}-{max_value})", 
        value=str(default_value)
    )
    try:
        input_value = float(input_value)
        if input_value < min_value or input_value > max_value:
            st.sidebar.warning(f"Please enter a value between {min_value} and {max_value} for {label}.")
            return None
        return input_value
    except ValueError:
        st.sidebar.warning(f"Please enter a valid number for {label}.")
        return None

employee_id = get_numerical_input('Employee_id', 1000,9999,1000)
no_of_trainings = get_numerical_input('Number of Trainings', 0, 10, 2)
age = get_numerical_input('Age', 20, 60, 30)
previous_year_rating = get_numerical_input('Previous Year Rating', 0.0, 5.0, 3.0)
length_of_service = get_numerical_input('Length of Service (Years)', 0, 20, 5)
awards_won = get_numerical_input('Awards Won', 0, 5, 0)
avg_training_score = get_numerical_input('Average Training Score', 50, 100, 75)

# Check for valid inputs
if None in [employee_id,no_of_trainings, age, previous_year_rating, length_of_service, awards_won, avg_training_score]:
    st.sidebar.error("Please correct invalid inputs before proceeding.")

# Collect user input into a dictionary
user_input = {
    'employee_id': employee_id,
    'department': department,
    'region': region,
    'education': education,
    'gender': gender,
    'recruitment_channel': recruitment_channel,
    'no_of_trainings': no_of_trainings,
    'age': age,
    'previous_year_rating': previous_year_rating,
    'length_of_service': length_of_service,
    'awards_won': awards_won,
    'avg_training_score': avg_training_score
}

# Preprocess user input
def preprocess_input(user_input, scaler, label_encoders):
    for col in label_encoders.keys():
        if user_input[col] not in label_encoders[col].classes_:
            user_input[col] = label_encoders[col].classes_[0]  # Default to the first class for unseen values

    user_df = pd.DataFrame([user_input])
    for col in label_encoders.keys():
        user_df[col] = label_encoders[col].transform(user_df[col])
    scaled_features = scaler.transform(user_df)
    return scaled_features

# Feature importance explanation
def generate_detailed_explanation(user_input, prediction_probability, feature_weights):
    top_features = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)[:5]
    explanation = f"Prediction: The employee's likelihood of promotion is {prediction_probability:.2f}.\n\n"
    explanation += "Key Factors Influencing the Prediction:\n"
    
    for feature, weight in top_features:
        value = user_input.get(feature, 'N/A')
        explanation += f"- {feature.capitalize()}: {value} (Importance: {weight:.2f})\n"

    explanation += "\nConclusion:\n"
    explanation += (
        f"The overall prediction suggests a {'high' if prediction_probability > 0.75 else 'moderate' if prediction_probability > 0.5 else 'low'} likelihood of promotion."
    )
    return explanation

# Feature weight extraction
def get_feature_weights(model, feature_names):
    first_layer_weights = model.layers[0].get_weights()[0]
    average_weights = np.mean(np.abs(first_layer_weights), axis=1)
    feature_importance = dict(zip(feature_names, average_weights))
    sorted_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    return sorted_features

# Prediction button
if st.sidebar.button('Predict Promotion'):
    # Preprocess input
    preprocessed_input = preprocess_input(user_input, scaler, label_encoders)
    prediction_probability = model.predict(preprocessed_input)[0][0]

    # Get feature weights
    feature_names = list(user_input.keys())
    feature_weights = get_feature_weights(model, feature_names)

    # Generate explanation
    explanation = generate_detailed_explanation(user_input, prediction_probability, feature_weights)

    # Display results
    st.subheader("Prediction Results")
    st.write(f"Promotion Likelihood: {prediction_probability:.2f}")
    st.subheader("Detailed Explanation")
    st.write(explanation)
