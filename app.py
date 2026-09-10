# Main Streamlit app combining everything
import streamlit as st
from student_view import show_student_view
from tpo_dashboard import show_tpo_dashboard

st.set_page_config(page_title="AI Placement Predictor", layout="wide")

# Simple navigation
page = st.sidebar.selectbox("Navigate", ["Student View", "TPO Dashboard"])

if page == "Student View":
    show_student_view()
else:
    show_tpo_dashboard()
