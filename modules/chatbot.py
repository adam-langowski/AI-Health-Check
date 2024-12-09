import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import os
from utils.chatbot_helper import *
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def app():
    st.title("Welcome to the AI Health Check App")
    with open("styles/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    load_dotenv()
    PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

    # Loading embeddings and Pinecone index
    embeddings = download_hugging_face_embeddings()
    index_name = "medical-assistant"    

    # Initializing Pinecone vector store 
    doc_search = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retriever = doc_search.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # User input 
    user_question = st.text_input("Enter your medical question:")
    if user_question:
        # Initializing Ollama LLM
        llm = OllamaLLM(model="llama3.2")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", chatbot_system_prompt),
                ("human", "{input}"),
            ]
        )
        
        # information retrieval
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": user_question})

        # model's response
        st.write(response["answer"])