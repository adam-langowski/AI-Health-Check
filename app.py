import streamlit as st

st.set_page_config(
    page_title="AI Health Check",
    layout='wide',
    page_icon='🩺'
)
with open("styles/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from modules import diabetes, brain_tumor, blood_cells, organ_segmentation, retinal_vessels, pneumonia, chatbot
pages = {
    "🏠 Home - Medical Assistant": chatbot.app,
    "🔬 Blood Cells Detection": blood_cells.app,
    "🧠 Brain Tumor Detection": brain_tumor.app,
    "🩸 Diabetes Prediction": diabetes.app,
    "🫀 Organ Segmentation": organ_segmentation.app,
    "🫁 Pneumonia Detection": pneumonia.app,
    "👁️ Retinal Vessel Segmentation": retinal_vessels.app,
}

st.sidebar.markdown(
        """
        <div class="sidebar-header">
            <h1>AI Health Check 🩺</h1>
            <p style="font-size:20px; color:dark-gray;">Your health assistant powered by AI</p>
        </div>
        """, unsafe_allow_html=True)

selected_module = st.sidebar.radio('', options=list(pages.keys()), label_visibility="collapsed")

st.sidebar.markdown('<div class="custom-header">Filters:</div>', unsafe_allow_html=True)

pages[selected_module]()

st.markdown(
"""
<hr>
<p style='text-align: center; color: gray;'>© 2024 AI Health Check. All rights reserved.</p>
""", unsafe_allow_html=True)