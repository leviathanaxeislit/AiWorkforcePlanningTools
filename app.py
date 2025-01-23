import streamlit as st

st.set_page_config(
    page_title="Ai Workforce Planning Tool",
    page_icon="https://www.commercient.com/wp-content/uploads/2019/12/deepLearning.gif",
    initial_sidebar_state="collapsed",
)
with open("styles/style.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)


# Main page
st.markdown("""
            <p style="font-size: 35px; font-family: 'Orbitron', sans-serif;font-weight: bold; color: yellow;">AI WORKFORCE PLANNING TOOLS</p>
            """,unsafe_allow_html=True)
st.image(
    image="https://gifdb.com/images/high/ai-finger-print-recognition-zl4ku51ojamo22k9.gif"
)
st.markdown(
    """
            AI Workforce Planning Tools is an advanced **AI-powered system** designed to enhance workforce management by utilizing machine learning models for **promotion prediction** and **role recommendation**. By leveraging AI, the tool helps organizations make informed decisions about employee promotions and match resumes with the most suitable roles, improving workforce efficiency and planning.

The project includes:
- **Promotion Prediction Model**: Predicts employee promotions using a TensorFlow-based neural network model.
- **Role Recommendation System**: Uses collaborative filtering to recommend role based on resume and job embeddings."""
)

st.write("Use the sidebar to navigate to different tools.")
