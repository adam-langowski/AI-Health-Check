import streamlit as st
from modules import diabetes, brain_tumor, blood_cells, organ_segmentation, retinal_vessels

st.set_page_config(
    page_title="AI Health Check",
    layout='wide',
    page_icon='🩺'
)

pages = {
    "Blood Cells Detection": blood_cells.app,
    "Brain Tumor Detection": brain_tumor.app,
    "Diabetes Prediction": diabetes.app,
    "Organ segmentation": organ_segmentation.app,
    "Retinal Vessel Segmentation": retinal_vessels.app
}

with open("styles/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown('<div class="custom-header">Available modules:</div>', unsafe_allow_html=True)
selected_module = st.sidebar.radio('', options=list(pages.keys()))

st.sidebar.markdown('<div class="custom-header">Filters:</div>', unsafe_allow_html=True)

pages[selected_module]()