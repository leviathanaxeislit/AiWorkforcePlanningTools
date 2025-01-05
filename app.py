import streamlit as st

st.set_page_config(
    page_title="Ai Workforce Planning Tool",
    page_icon="https://cdn2.iconfinder.com/data/icons/artificial-intelligence-6/64/ArtificialIntelligence19-512.png",
    initial_sidebar_state="collapsed",
    
)
with open("styles/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


# Main page
st.title("Ai WorkForce Planning Tools")
st.markdown(
    """
            AI Workforce Planning Tools is an advanced **AI-powered system** designed to enhance workforce management by utilizing machine learning models for **promotion prediction** and **role recommendation**. By leveraging AI, the tool helps organizations make informed decisions about employee promotions and match resumes with the most suitable roles, improving workforce efficiency and planning.

The project includes:
- **Promotion Prediction Model**: Predicts employee promotions using a TensorFlow-based neural network model.
- **Role Recommendation System**: Uses collaborative filtering to recommend role based on resume and job embeddings."""
)

st.write("Use the sidebar to navigate to different tools.")
