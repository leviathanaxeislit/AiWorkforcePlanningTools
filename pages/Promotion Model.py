import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import Sequential, load_model

st.set_page_config(
    page_title="Promotion Model",
    page_icon="https://cdn2.iconfinder.com/data/icons/knowledge-promotion-3/64/career_leadership_learn_development_growth_motivation-256.png",
    # initial_sidebar_state="expanded",  #Optional: remove or uncomment as needed.
)

# --- Navigation Bar ---
cols = st.columns(3)  # Create 3 columns

try:
    with cols[0]:
        st.page_link(page="app.py", icon="🏠", label="Home")
    with cols[1]:
        st.page_link(page="pages/Promotion Model.py", icon="💹", label="Promotion Model")
    with cols[2]:
        st.page_link(page="pages/Role Recommendation Model.py", icon="🏢", label="Role Recommendation")
except Exception as e:
    st.error(f"Error loading page links: {e}")

st.markdown(
    """
            <p style="font-size: 40px; font-family: 'Gugi', serif;font-weight: 400;">EMPLOYEE PROMOTION PREDICTION</p>
            """,
    unsafe_allow_html=True,
)
st.image(
    image="https://pulsemotivation.com/wp-content/uploads/2022/02/Pulse-Motivation-Employee-engagement-strategy%E2%80%A8.gif",
    use_container_width="auto",
    caption="Employee Promotion Model",
)

with open("styles/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)

# --- Model Loading and Preprocessing ---
model_path = "models/employee_promotion_model.h5"
try:
    model = load_model(model_path)
    scaler = joblib.load("models/scaler.pkl")
    label_encoders = joblib.load("models/label_encoders.pkl")
except FileNotFoundError:
    st.error("Error: Model files not found. Please ensure 'models' folder exists and contains the necessary files.")
    st.stop()  # Stop execution if model files are missing
except Exception as e:
    st.error(f"An error occurred loading the model: {e}")
    st.stop()


# --- Input Fields (Now in Main Content) ---
st.write(
    "Use this tool to predict the likelihood of an employee being promoted based on their profile and performance data."
)

st.markdown("""
            <p style="font-size: 35px;font-weight: bold;">  Input Employee Details</p>
            """,unsafe_allow_html=True)


def get_numerical_input(label, min_value, max_value, default_value):
    input_value = st.text_input(
        f"{label} (Range: {min_value}-{max_value})", value=str(default_value)
    )
    try:
        input_value = float(input_value)
        if input_value < min_value or input_value > max_value:
            st.warning(
                f"Please enter a value between {min_value} and {max_value} for {label}."
            )
            return None
        return input_value
    except ValueError:
        st.warning(f"Please enter a valid number for {label}.")
        return None


# Input fields (moved to main content)
employee_id = get_numerical_input("Employee ID", 1000, 9999, 1000)
department = st.selectbox(
    "Department",
    [
        "Sales & Marketing",
        "Operations",
        "Technology",
        "Analytics",
        "R&D",
        "Procurement",
        "Finance",
        "HR",
        "Legal",
    ],
)
region = st.selectbox(
    "Region",
    [
        "Bangalore",
        "Hyderabad",
        "Pune",
        "Chennai",
        "Mumbai",
        "Delhi",
        "Noida",
        "Gurgaon",
        "Kolkata",
        "Ahmedabad",
        "Jaipur",
        "Lucknow",
        "Kochi",
        "Thiruvananthapuram",
        "Indore",
        "Bhubaneswar",
        "Nagpur",
        "Visakhapatnam",
        "Chandigarh",
        "Coimbatore",
        "Mysore",
        "Vadodara",
        "Patna",
        "Ranchi",
        "Guwahati",
        "Surat",
        "Hubli",
        "Jamshedpur",
        "Dehradun",
        "Raipur",
        "Amritsar",
        "Pondicherry",
        "Shillong",
        "Shimla",
    ],
)
education = st.selectbox(
    "Education Level",
    ["Master's & above", "Bachelor's", "Below Secondary", "Master's", "High School"],
)
gender = st.selectbox("Gender", ["m", "f"])
recruitment_channel = st.selectbox(
    "Recruitment Channel",
    ["linkedin", "sourcing", "other", "Naukri", "Indeed", "referred"],
)
no_of_trainings = get_numerical_input("Number of Trainings", 0, 10, 2)
age = get_numerical_input("Age", 20, 60, 30)
previous_year_rating = get_numerical_input("Previous Year Rating", 0.0, 5.0, 3.0)
length_of_service = get_numerical_input("Length of Service (Years)", 0, 20, 5)
awards_won = get_numerical_input("Awards Won", 0, 5, 0)
avg_training_score = get_numerical_input("Average Training Score", 50, 100, 75)


# Check for valid inputs
if None in [
    employee_id,
    no_of_trainings,
    age,
    previous_year_rating,
    length_of_service,
    awards_won,
    avg_training_score,
]:
    st.error("Please correct invalid inputs before proceeding.")
    st.stop()

# --- Data Preprocessing and Prediction ---

user_input = {
    "employee_id": employee_id,
    "department": department,
    "region": region,
    "education": education,
    "gender": gender,
    "recruitment_channel": recruitment_channel,
    "no_of_trainings": no_of_trainings,
    "age": age,
    "previous_year_rating": previous_year_rating,
    "length_of_service": length_of_service,
    "awards_won": awards_won,
    "avg_training_score": avg_training_score,
}

def preprocess_input(user_input, scaler, label_encoders):
    for col in label_encoders.keys():
        if user_input[col] not in label_encoders[col].classes_:
            user_input[col] = label_encoders[col].classes_[0]

    user_df = pd.DataFrame([user_input])
    for col in label_encoders.keys():
        user_df[col] = label_encoders[col].transform(user_df[col])
    scaled_features = scaler.transform(user_df)
    return scaled_features


def generate_detailed_explanation(user_input, prediction_probability, feature_weights):
    top_features = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)[:5]
    explanation = f"Prediction: The employee's likelihood of promotion is {prediction_probability:.2f}.\n\n"
    explanation += "Key Factors Influencing the Prediction:\n"

    for feature, weight in top_features:
        value = user_input.get(feature, "N/A")
        explanation += f"- {feature.capitalize()}: {value} (Importance: {weight:.2f})\n"

    explanation += "\nConclusion:\n"
    explanation += f"The overall prediction suggests a {'high' if prediction_probability > 0.75 else 'moderate' if prediction_probability > 0.5 else 'low'} likelihood of promotion."
    return explanation


def get_feature_weights(model, feature_names):
    first_layer_weights = model.layers[0].get_weights()[0]
    average_weights = np.mean(np.abs(first_layer_weights), axis=1)
    feature_importance = dict(zip(feature_names, average_weights))
    sorted_features = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )
    return sorted_features


# --- Prediction Button and Results ---
if st.button("Predict Promotion"):
    try:
        preprocessed_input = preprocess_input(user_input, scaler, label_encoders)
        prediction_probability = model.predict(preprocessed_input)[0][0]
        feature_names = list(user_input.keys())
        feature_weights = get_feature_weights(model, feature_names)
        explanation = generate_detailed_explanation(
            user_input, prediction_probability, feature_weights
        )

        st.subheader("Prediction Results")
        st.write(f"Promotion Likelihood: {prediction_probability:.2f}")
        st.subheader("Detailed Explanation")
        st.write(explanation)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")