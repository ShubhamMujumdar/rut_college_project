import os
from pdf_utils import load_pdf_from_path
from chunk import chunk_text
from embed_store import embed_store

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    model="qwen/qwen3-32b", 
    api_key=" "
)

def rag_pipeline(pdf_path, query, k=3):
    raw_text = load_pdf_from_path(pdf_path)
    raw_text = " ".join([doc.page_content for doc in raw_text])
    chunks = chunk_text(raw_text)
    vectorstore = embed_store(chunks)

    relevant_docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""You are a highly intelligent chatbot assistant that answers questions based on relevant context from a document uploaded by the user.
                Context: {context}
                Question: {query}
                Answer:
               """

    response = llm.invoke(prompt)

    return response.content

if __name__ == "__main__":
    pdf_path = "./DAA_2024.pdf"
    query = "What are the names of the textbooks mentioned?"

    if not os.path.exists(pdf_path):
        print(f"PDF not found at path: {pdf_path}")
    else:
        answer = rag_pipeline(pdf_path, query)
        print("\nAnswer: ", answer)