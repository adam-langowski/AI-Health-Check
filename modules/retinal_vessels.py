import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

# MODELS
@st.cache_resource
def load_models():
    """Wczytaj modele z zasobów aplikacji"""
    models = {
        "HRF": tf.keras.models.load_model('models/model_segmentingVessels.keras'),
        "DRIVE": tf.keras.models.load_model('models/model_DRIVE_segmentingVessels.keras')
    }
    return models

models = load_models()

def app():
    st.title("Retinal Blood Vessel Segmentation")

    st.markdown("<p style='text-align: center; color: white; font-size: 18px;'>Upload an image and select a model to segment retinal blood vessels</p>", unsafe_allow_html=True)

    # Sidebar for model selection
    st.sidebar.markdown("<h2 style='color: dark-grey;'>Model Selection</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='color: dark-gray; font-size: 16px;'>There are 2 models available, trained on 2 different databases.</p>", unsafe_allow_html=True)

    model_choice = st.sidebar.selectbox(
        "Choose an image format (database):",
        ("HRF", "DRIVE")
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    # Segmenting process
    if uploaded_file is not None:
        st.markdown("<h5 style='color: #50586C;'>Original Image:</h5>", unsafe_allow_html=True)
        st.image(uploaded_file, use_column_width=True)

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        model = models.get(model_choice, models['HRF'])

        with st.spinner("Segmenting..."):
            prediction = segment_image(image, model)
            segmented_image = apply_threshold(prediction)

        st.markdown("<h5 style='color: #50586C;'>Segmented Image:</h5>", unsafe_allow_html=True)
        st.image(segmented_image, use_column_width=True)

def segment_image(image, model):
    target_height = 584
    target_width = 876
    resized_image = cv2.resize(image, (target_width, target_height))
    resized_image = np.expand_dims(resized_image, axis=0) / 255.0
    prediction = model.predict(resized_image)
    return prediction[0]

def apply_threshold(pred, threshold=0.42):
    binary_image = np.where(pred >= threshold, 255, 0).astype(np.uint8)
    return binary_image
