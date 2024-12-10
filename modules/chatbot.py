import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import os
from utils.chatbot_helper import *
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import pinecone
import socket

def app():
    st.title("Welcome to the AI Health Check App")
    with open("styles/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    load_dotenv()
    # Initialize environment variable for Pinecone
    try:
        PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    except KeyError:
        st.error("PINECONE_API_KEY not found. Please log in to Pinecone and set the environment variable. "
                 "\nMore instructions in README.md")
    
    # Loading embeddings and Pinecone index
    try:
        embeddings = download_hugging_face_embeddings()
        index_name = "medical-assistant"    

        # Initializing Pinecone vector store 
        doc_search = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        retriever = doc_search.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    except pinecone.exceptions.PineconeConnectionError as e:
        st.error(
            "Error connecting to Pinecone. Please ensure you are logged in to Pinecone and have set the PINECONE_API_KEY environment variable. "
            "\nAdditionally, run the script `utils/store_index.py` to create embeddings and store them in your Pinecone index."
        )
        return
    except Exception as e:
        st.error(f"An error occurred while setting up Pinecone: {e}. \nPlease ensure you are logged in to Pinecone and have set the PINECONE_API_KEY environment variable. "
            "\nAdditionally, run the script `utils/store_index.py` to create embeddings and store them in your Pinecone index.")
        return
    
    # User input 
    user_question = st.text_input("Enter your medical question:")
    if user_question:
        try:
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
        except socket.error as e:
            st.error(
                "Error connecting to Ollama model. Please ensure that Ollama is installed and the command 'ollama run llama3.2' is running. "
                "\nMore instructions in README.md."
            )
        except Exception as e:
            st.error(f"An error occurred while processing the question: {e}. "
                    "\nPlease ensure that Ollama is installed and the command 'ollama run llama3.2' is running. "
                    "\nMore instructions in README.md.")