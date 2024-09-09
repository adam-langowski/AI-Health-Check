import streamlit as st
import pickle
import numpy as np

def app():
    st.title("Diabetes Prediction 🩸")

    # MODEL
    diabetes_model = pickle.load(open('models/model_diabetes.pkl', 'rb'))

    # INPUTS
    col1, col2, col3 = st.columns(3)
    with col1: 
        pregnancies = st.text_input("Number of pregnancies") 
    with col2:
        glucose = st.text_input("Glucose level")
    with col3:
        blood_pressure = st.text_input("Blood pressure")
    with col1: 
        skin_thickness = st.text_input("Skin thickness") 
    with col2:
        insulin = st.text_input("Insulin level")
    with col3:
        bmi = st.text_input("BMI")
    with col1: 
        pedigree_function = st.text_input("Diabetes pedigree function value") 
    with col2:
        age = st.text_input("Age")

    diabetes_result = ""
    
    # PREDICTION
    if st.button("Prediction:"):
        user_input = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree_function, age]
        
        # INPUTS VALIDATION
        if all(user_input) and all(x.replace('.', '', 1).isdigit() for x in user_input):  
            input_data = np.array([float(x) for x in user_input]).reshape(1, -1)  # Reshape to 2D array
            
            prediction = diabetes_model.predict(input_data)
            
            if prediction[0] == 1:
                diabetes_result = "Diabetes: POSITIVE"
                st.error(diabetes_result)

            elif prediction[0] == 0:
                diabetes_result = "Diabetes: NEGATIVE"
                st.success(diabetes_result)
        else:
            st.error("Please fill out all fields with numeric values before making a prediction.")
