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
            AI Workforce Planning Tools is an advanced **AI-powered system** designed to enhance workforce management by utilizing machine learning models for **promotion prediction** and **role recommendation**. By leveraging AI, the tool helps organizations make informed decisions about employee promotions and match resumes with the most suitable roles, improving workforce efficiency and planning.

The project includes:
- **Promotion Prediction Model**: Predicts employee promotions using a TensorFlow-based neural network model.
- **Role Recommendation System**: Uses collaborative filtering to recommend role based on resume and job embeddings."""
)

st.write("Use the navigation bar  to navigate to different tools.")
