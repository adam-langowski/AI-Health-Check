'''
RUNNING THIS FILE CREATES EMBEDDINGS FROM PDF DATA GIVEN IN data/chatbot_data
AND STORES THEM IN PINECONE INDEX
'''

from chatbot_helper import load_pdf_docs, split_data, download_hugging_face_embeddings
import os
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_docs(data='data/chatbot_data/')
chunks = split_data(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-assistant"

pc.create_index(
    name=index_name,
    dimension=384,   # model dimensions
    metric="cosine", # model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

doc_search = PineconeVectorStore.from_documents(
    documents=chunks,
    index_name=index_name,
    embedding=embeddings
)