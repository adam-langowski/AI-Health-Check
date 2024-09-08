import streamlit as st
import pickle

def app():
    st.title("Diabetes Prediction")
    diabetes_model = pickle.load(open('models/model_diabetes.pkl', 'rb'))
