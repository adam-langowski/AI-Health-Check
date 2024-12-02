import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
from utils import helpers

def app():
    st.title("Pneumonia Detection 🫁")

    with open("styles/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # load model
    @st.cache_resource
    def load_pneumonia_model():
        return load_model("models/pneumonia_detection_model.h5")

    pneumonia_model = load_pneumonia_model()

    # select file
    uploaded_file = st.file_uploader("**Choose a lungs X-ray image**", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown(
            '<img class="center-image" src="data:image/jpeg;base64,{}" width="300"/>'.format(
                helpers.image_to_base64(image)
            ),
            unsafe_allow_html=True,
        )

        if st.button('Pneumonia Detection'):
            # Preprocessing 
            image = np.array(image)
            if len(image.shape) == 2:  #RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = cv2.resize(image, (128, 128))
            image = image / 255.0  
            image = np.expand_dims(image, axis=0)  # Batch dimension

            # Prediction
            prediction = pneumonia_model.predict(image)

            if prediction[0][0] > 0.5:
                st.error("Pneumonia Detected")
            else:
                st.success("No Pneumonia Detected")