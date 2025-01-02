import streamlit as st
from PIL import Image
from collections import Counter
from ultralytics import YOLO
from utils import helpers

def app():
    st.title("Blood Cell Detection 💦")
    
    # MODEL
    @st.cache_resource
    def load_model():
        model_path = 'models/model_blood_cells_best.pt'  
        model = YOLO(model_path)                
        return model 
    
    with open("styles/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Confidence level slider
    confidence_level = st.slider("**Confidence level:**", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    
    # Uploading img
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # INPUT IMG
        image = Image.open(uploaded_file)

        if st.button("Detect cells"):

            model = load_model()

            # PREDICTION
            annotated_image, cell_count = predict_and_annotate(image, model, confidence_level)
            
            annotated_image_base64 = helpers.image_to_base64(annotated_image)
            st.markdown(
                f'<img class="center-image" src="data:image/png;base64,{annotated_image_base64}" width="640">',
                unsafe_allow_html=True
            )
            
            # Display detected cells in a table
            st.write("Detected cells:")
            st.markdown(helpers.generate_html_table(cell_count), unsafe_allow_html=True)

# Prediction and results 
def predict_and_annotate(image, model, confidence):
    # prediction
    result = model.predict(source=image, imgsz=640, conf=confidence)
    annotated_res = result[0].plot()  

    # detection data
    detection = result[0].boxes.data
    class_names = [model.names[int(cls)] for cls in detection[:, 5]]  
    count = Counter(class_names)

    annotated_image = Image.fromarray(annotated_res) # PIL img

    return annotated_image, count