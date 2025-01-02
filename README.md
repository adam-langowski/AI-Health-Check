# AI-Health-Check

**Multimodel Machine Learning Application**

**Explore the app:** [ai-health-check.streamlit.app](https://ai-health-check.streamlit.app/)

---

## Sample Data

You can download sample data for the application here: [sample data](https://github.com/adam-langowski/AI-Health-Check/tree/main/data/sample%20data)

---

## Key Features and Tools

| **Feature**                        | **Technologies/Tools**                                           | **Dataset**                                                                                           |
| ---------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Diabetes Prediction**            | `scikit-learn`, `numpy`, `pandas`, `PCA`, `VotingClassifier` | [Diabetes Dataset (Kaggle)](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)                         |
| **Brain Tumor Detection**          | `TensorFlow`, `Keras`, `OpenCV`, `PIL`, `seaborn`            | [Brain Tumor Dataset (Kaggle)](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)             |
| **Blood Cell Detection and Count** | `PyTorch`, `YOLOv10`, `PIL`, `OpenCV`                          | [Blood Cells Dataset (Roboflow)](https://universe.roboflow.com/clg-vtj9f/blood-cell-detection-bsbvn/dataset/4) |
| **Liver Segmantation**             | `PyTorch`, `MONAI`, `3D slicer`, `NIfTI`, `DICOM`            | [Liver Dataset (The Medical Segmentation Decathlon)](http://medicaldecathlon.com/)                             |
| **Organ Segmantation**             | `PyTorch`, `MONAI`, `tcia`, `DICOM`, `torch`                 | [Organs dataset (The Cancer Imaging Archive)](https://www.cancerimagingarchive.net/)                           |
| **Retinal Vessels Segmantation**   | `Keras`, `UNet`, `OpenCV`                                        | [Full repo about this project](https://github.com/adam-langowski/Segmenting-Blood-Vessels-With-CNN/)           |
| **Pneumonia Detection**            | `TensorFlow`, `Keras`, `albumentations`, `OpenCV`              | [Pneumonia Dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)           |
| **Medical Chatbot**                | `Pinecone Vector Database`, `RAG`, `LLM`, `langchain`          | `Gale Encyclopedia Of Medicine Vol2`                                                                      |

---

## Instructions

Follow these steps to run the application locally and use all of the functionalities (chatbot can only run locally):

### 1. Install **Ollama** and run the Llama model

- First, make sure that **Ollama** is installed on your machine. You can download it from the official website: [https://ollama.com/](https://ollama.com/).
- Once Ollama is installed, open a terminal and run the following command to start the Llama model (first usage will download the model):

  ```bash
  ollama run llama3.2
  ```

### 2. Install dependencies

- Install the necessary Python dependencies for the project by running:

  ```bash
  pip install -r requirements.txt
  ```

### 3. Set up Pinecone

- Create a Pinecone account by going to https://www.pinecone.io/.
- After signing up, get your Pinecone API Key from your account dashboard.
- Create a .env file in the root directory of the project and add the following line with your Pinecone API Key:

PINECONE_API_KEY="*your_pinecone_api_key*"

- Now, run the script utils/store_index.py to create embeddings and store them in the Pinecone index:

  ```bash
  python utils/store_index.py
  ```

This will initialize and populate the Pinecone index with necessary data for the application.

### 4. Run the application

- Finally start the application by running:

  ```bash
  streamlit run app.py
  ```

This will launch the app in your browser.
