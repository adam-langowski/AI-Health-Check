import streamlit as st
from PIL import Image
import os
from utils import helpers

def app():
    st.title("Organ Segmentation Simulation")

    with open("styles/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
    # Message
    st.markdown(""" 
    **Note:**  
    **Running the organ segmentation model is computationally expensive and involves extensive pre- and post-processing steps.**  
    **The process can take considerable time and may be prone to errors.** **Below, we simulate the input and output of the model.**  
    **For the actual implementation and tests, refer to the `organ_segmentation.ipynb` notebook.**
    """)
    
    # Buttons side by side
    col1, col2 = st.columns([1, 1]) 

    with col1:
        if st.button("Read NIfTI"):
            st.session_state.display_image = 'input'
    
    with col2:
        if st.button("Segment all organs"):
            st.session_state.display_image = 'output'

    # Display images
    if 'display_image' in st.session_state:
        if st.session_state.display_image == 'input':
            try:
                input_image_path = os.path.join(os.getcwd(), 'assets/images/CT_coronal_slice.png')
                input_image = Image.open(input_image_path)
                st.markdown(
                    f'<img src="data:image/png;base64,{helpers.image_to_base64(input_image)}" class="center-image" width="200"/>',
                    unsafe_allow_html=True
                )
                st.markdown('<p class="center-caption">Simulated Input: NIfTI Image</p>', unsafe_allow_html=True)
            except FileNotFoundError:
                st.error("Input image not found. Ensure the image exists at the specified path.")
        elif st.session_state.display_image == 'output':
            try:
                output_image_path = os.path.join(os.getcwd(), 'assets/images/segmentation_coronal_slice.png')
                output_image = Image.open(output_image_path)
                st.markdown(
                    f'<img src="data:image/png;base64,{helpers.image_to_base64(output_image)}" class="center-image" width="400"/>',
                    unsafe_allow_html=True
                )
                st.markdown('<p class="center-caption">Simulated Output: Segmented Organs</p>', unsafe_allow_html=True)
            except FileNotFoundError:
                st.error("Output image not found. Ensure the image exists at the specified path.")
