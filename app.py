import streamlit as st
from modules import diabetes, brain_tumor, blood_cells

st.set_page_config(
    page_title="AI Health Check",
    layout='wide',
    page_icon='🩺'
)

pages = {
    "Diabetes Prediction": diabetes.app,
    "Brain Tumor Detection": brain_tumor.app,
    "Blood Cells Detection": blood_cells.app,
}

with open("styles/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown('<div class="custom-header">Available modules:</div>', unsafe_allow_html=True)
selected_module = st.sidebar.radio('', options=list(pages.keys()))
pages[selected_module]()