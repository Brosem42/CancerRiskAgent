import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="CancerRiskAgent", layout="centered")
st.title("CancerRiskAgent (NICE NG12)")

tab1, tab2 = st.tabs(["Assess patient", "Chat"])

with tab1:
    st.subheader("Assess a patient by ID")
    patient_id = st.text_input("Patient ID", value="PT-110")
    top_k = st.slider("Top K evidence", 1, 20, 8)

    if st.button("Run assessment"):
        r = requests.post(f"{API_URL}/assess", json={"patient_id": patient_id, "top_k": top_k})
        st.write("Status:", r.status_code)
        st.json(r.json())

with tab2:
    st.subheader("Grounded chat")
    session_id = st.text_input("Session ID", value="demo")
    msg = st.text_area("Message", value="What does NG12 say about persistent cough in smokers aged 40+?")
    top_k_chat = st.slider("Top K evidence (chat)", 1, 20, 6)

    if st.button("Send"):
        r = requests.post(
            f"{API_URL}/chat",
            json={"session_id": session_id, "message": msg, "top_k": top_k_chat},
        )
        st.write("Status:", r.status_code)
        st.json(r.json())