import streamlit as st
from PIL import Image
from collections import Counter
from ultralytics import YOLO
from utils import helpers

def app():
    st.title("Blood Cell Detection 💦")
    
    with open("styles/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # INPUT IMG
        image = Image.open(uploaded_file)
                
        # MODEL
        model_path = '../models/model_blood_cells_best.pt'  
        model = YOLO(model_path)
        
        # PREDICTION
        annotated_image, cell_count = predict_and_annotate(image, model)
        
        # DISPLAY RESULT
        st.image(annotated_image, width=640, caption='Annotated Image (640x640)', use_column_width=False)
        
        st.write("Detected Cells:")
        st.markdown(helpers.generate_html_table(cell_count), unsafe_allow_html=True)


# prediction and results 
def predict_and_annotate(image, model):
    # prediction
    result = model.predict(source=image, imgsz=640, conf=0.25)
    annotated_res = result[0].plot()  

    # detection data
    detection = result[0].boxes.data
    class_names = [model.names[int(cls)] for cls in detection[:, 4]]  
    count = Counter(class_names)

    return annotated_res, count