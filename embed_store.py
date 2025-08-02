from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def embed_store(chunks, persist_directory="chroma_db"):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    vectordb.persist()

    return vectordb

