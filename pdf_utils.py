import os
from langchain_community.document_loaders import PyMuPDFLoader

def save_uploaded_file(uploaded_file, save_dir="uploaded_pdfs"):

    #saves an uploaded file (from streamlit) to disk
    #returns the file path

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)

    with open(file_path,"wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def load_pdf_from_path(path):

    #loads a pdf and returns list of document objects
    loader = PyMuPDFLoader(path)
    return loader.load()