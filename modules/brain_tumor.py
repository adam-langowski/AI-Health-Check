import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np
import cv2
from utils import helpers

def app():
    st.title("Brain Tumor Detection 🧠")
    with open("styles/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
    # MODEL
    brain_model = keras.models.load_model('models/brain_model_best.keras')

    # IMAGE UPLOAD
    uploaded_file = st.file_uploader("Choose a brain MRI image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown('<img class="center-image" src="data:image/jpeg;base64,{}" width="300"/>'.format(
            helpers.image_to_base64(image)), unsafe_allow_html=True)        
        
        if st.button('Brain Tumor Detection'):
            # IMAGE PREPROCESSING
            image = np.array(image)  
            image = cv2.resize(image, (64, 64))  
            image = Image.fromarray(image, 'RGB')  
            image = np.array(image)  
            image = image / 255.0  
            image = np.expand_dims(image, axis=0)  # batch dimension

            # PREDICTION
            prediction = brain_model.predict(image)

            # RESULT
            if prediction[0][0] > 0.5:
                st.error("Tumor Detected")
            else:
                st.success("No Tumor Detected")
