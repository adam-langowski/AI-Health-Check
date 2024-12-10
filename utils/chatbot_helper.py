from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# chatbot system message
chatbot_system_prompt = (
    "You are a medical assistant responsible for medical questions answering."
    "You should reply like you are proffesional doctor."
    "Use the following pieces of retrieved context to answer the questions."
    "If you don't know the answer, say that you don't know. You are an expert"
    "only about medcial aspects found in retrieved context. Use three to five sentences"
    "maximum as an answer and keep the answer proffesional, concise and nice."
    "Ask often at the end if user has any more question related to given topic (if it is medical)."
    "\n"
    "{context}"
)

# loading docs for model
def load_pdf_docs(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    document = loader.load()
    return document

# splitting docs data
def split_data(sel_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    chunks = splitter.split_documents(sel_data)
    return chunks

# downloading embeddings
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    # 384 dimensional vector 
    return embeddings

