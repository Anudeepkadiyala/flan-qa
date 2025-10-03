import streamlit as st
import requests
import os

st.set_page_config(page_title="FLAN-T5 QA", page_icon="🤖")

st.title("🤖 FLAN-T5 QA (Fine-tuned on SQuAD)")
st.write("Enter a *context* passage and a *question*. The model will answer based on the context.")

api_url = st.text_input("API URL", value=os.getenv("API_URL", "http://localhost:8000/qa"))

context = st.text_area("Context", height=200, value="Alan Turing was a pioneering computer scientist who is widely considered the father of theoretical computer science and artificial intelligence.")
question = st.text_input("Question", value="Who was Alan Turing?")

if st.button("Get Answer"):
    with st.spinner("Querying model..."):
        try:
            resp = requests.post(api_url, json={"context": context, "question": question}, timeout=30)
            if resp.status_code == 200:
                st.success(resp.json().get("answer", ""))
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
