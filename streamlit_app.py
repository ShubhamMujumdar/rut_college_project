# PLEASE ADD COMMENTS EVERYWHERE THROUGHOUT THE PROJECT
import streamlit as st
from rag_pipeline import rag_pipeline
import re

st.set_page_config(page_title="RAG QA Assistant", layout="wide")

st.title("RAG QA Assistant")

uploaded_files = st.file_uploader("Upload your PDFs", type=['pdf'], accept_multiple_files=True)

docs = []

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} file(s)")

    with st.spinner("Embedding your documents..."):
        for file in uploaded_files:
            file_content = file.read()
            docs.append(file_content)

        st.success("Documents embedded and ready! Ask away.")

query = st.text_input("Ask a question")

if query:
    with st.spinner("Thinking..."):
        response = rag_pipeline(docs, query)

        clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        st.markdown("Response")
        st.write(clean_response)