import requests
import streamlit as st
from langchain_core.messages import HumanMessage
from ragPipeline.rag import retriever, graph, config
from scripts.document_loader import DocumentLoader
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

FASTAPI_URL = "http://127.0.0.1:8002" 

st.set_page_config(page_title="Cancer Risk Agentic Hub", page_icon=":robot:", layout="wide")

# init sesseion state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# when reruns occur, you can still keep the same chat history
for message in st.session_state.chat_history:
    print(f"message: {message}")
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# set docs for retriever to process when user uploads files
docs = retriever.add_uploaded_docs(st.session_state.uploaded_files)

#how retriever process human message
def process_message(message, history):
    """
    Takes user message + hisotry, sends it to FastAPI and returns the reply.
    """
    payload = {
        "message": message,
        "history": history
    }
    try:
        response = requests.post(f"{FASTAPI_URL}/chat", json=payload)
        response.raise_for_status()
        return response.json()["reply"]
    except Exception as e:
        return f"Error connecting to backend: {e}"

# this ignores the previous messages--TODO: change the prompt to provide access to previous messsages
st.markdown("""
# Cancer Risk Agentic Hub: Clinicial Decision Support
### *Your Personal AI Assistant for Cancer Risk Assessment*
""")

col1, col2 = st.columns([2, 1])

#column 1 
with col1:
    st.subheader("Chat Interface")
    #react to user input 
    if user_message:= st.chat_input("Enter your message:"):
        #display user message in chat container
        with st.chat_message("User"):
            st.markdown(user_message)
        #add user message
        response = process_message(user_message, st.session_state.chat_history)

        with st.chat_message("Clinical AI assistant"):
            st.markdown(response)

        #store both messages
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        response = process_message(user_message)

#column 2
with col2:
    st.subheader("Patient Data Management Hub")
    #file upload
    uploaded_files_uploader = st.file_uploader(
        "Upload Patient order requisition or enter patient MRN number",
        type=["pdf", "json"],
        accept_multiple_files=True
    )
    if uploaded_files_uploader:
        for file in uploaded_files_uploader:
            if file.name not in [f.name for f in st.session_state.uploaded_files]:
                # send to fastapi
                files = {"file": (file.name, file.getvalue(), file.type)}
                requests.post(f"{FASTAPI_URL}/upload", files=files)
                st.session_state.uploaded_files.append(file)