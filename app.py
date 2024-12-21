import streamlit as st

with open( "styles/style.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
    

# Main page
st.title("AI WorkForce Planning Tools")
st.write("Welcome to the AI WorkForce Planning Tools! Use the sidebar to navigate to different tools.")

