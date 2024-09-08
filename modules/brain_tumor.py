import streamlit as st
from tensorflow import keras

def app():
    st.title("Brain Tumor Detection")
    brain_model = keras.models.load_model('models/brain_model_best.keras')
