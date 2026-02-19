# streamlit app for knowledge base
import streamlit as st
from langchain_core.messages import HumanMessage
from ragPipeline.rag import retriever, graph, config
from scripts.document_loader import DocumentLoader

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
def process_message(message):
    """
    Assistant response.
    """
    response = graph.invoke({"messages": HumanMessage(message)},
                            config=config)
    return response["messages"][-1].content

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
        st.session_state.chat_history.append({"role": "user", "content": "user_message"})
        response = process_message(user_message)

#column 2
with col2:
    st.subheader("Patient Data Management Hub")
    #file upload
    uploaded_files = st.file_uploader(
        "Upload Patient order requisition or enter patient MRN number",
        type=["pdf", "json"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(file)