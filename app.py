import streamlit as st

st.set_page_config(
    page_title="Ai Workforce Planning Tool",
    page_icon="https://www.commercient.com/wp-content/uploads/2019/12/deepLearning.gif",
    initial_sidebar_state="collapsed",
)
with open("styles/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


cols = st.columns(3)  # Create 3 columns

try:
    with cols[0]:
        st.page_link(page="app.py", icon="🏠", label="Home")
    with cols[1]:
        st.page_link(
            page="pages/Promotion Model.py", icon="💹", label="Promotion Model"
        )
    with cols[2]:
        st.page_link(
            page="pages/Role Recommendation Model.py",
            icon="🏢",
            label="Role Recommendation",
        )
except Exception as e:
    st.error(f"Error loading page links: {e}")

# Main page
st.markdown(
    """
            <p style="font-size: 40px; font-family: 'Gugi', serif;font-weight: 400;border-radius: 2px;">AI WORKFORCE PLANNING TOOLS</p>
            """,
    unsafe_allow_html=True,
)
st.image(
    image="https://gifdb.com/images/high/ai-finger-print-recognition-zl4ku51ojamo22k9.gif"
)


st.markdown(
    """
    ## 💡 Overview

AI Workforce Planning Tools is your go-to **AI-powered system** for making smarter workforce decisions! 🌟 Designed to enhance workforce management, it uses machine learning models for **promotion prediction** and intelligent **role recommendation**, now with magical ✨ automated cover letter generation! It helps organizations make data-driven decisions, match awesome talent with the right roles, and simplifies the job application process with AI-generated cover letters! 🚀 Improving workforce efficiency and strategic planning like never before!

What you'll find inside:

-   **Promotion Prediction Model**: 🏆 Accurately predicts employee promotions using a TensorFlow-based neural network model.
-   **Role Recommendation System**: 🏢 Intelligently recommends suitable roles based on comprehensive resume analysis and job embeddings.
-   **AI-Powered Cover Letter Generation:** ✍️ Automates the creation of tailored cover letters using Google's Gemini LLM, making job applications a breeze!

## ✨ Key Features

-   **Promotion Prediction**: 📈 Accurately predicts the likelihood of employee promotions based on key factors like department, performance ratings, training scores, and more!

-   **Intelligent Role Recommendations**: 🎯 Analyzes resumes and provides personalized job recommendations, with suitability scores and links to job postings on LinkedIn & Naukri. 💼

-   **AI-Driven Cover Letter Generation**: ✉️ Generates compelling cover letters tailored to specific job descriptions using Google's Gemini LLM. Say goodbye to writer's block! ✍️

-   **Resume Scoring**: 💯 Provides a suitability score for any resume and job description combination! Helping you assess fit at a glance.

-   **Streamlit Interface**: 💻 User-friendly web interface for easy access to all features and models!

-   **Advanced Model Integration**: 🧠 Utilizes state-of-the-art machine learning models built with TensorFlow and PyTorch.


            """
)

st.markdown("### Use the navigation bar  to navigate to different tools.")
