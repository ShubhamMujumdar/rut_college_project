# tools.py - CORRECTED VERSION
import streamlit as st
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
import os


def get_llm():
    """Get an LLM instance"""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )


@tool("evaluate_sufficiency_tool")
def evaluate_sufficiency(input_text: str) -> str:
    """
    Evaluates if documents are sufficient to answer the query.
    Input: 'QUERY: <question> DOCUMENTS: <documents>'
    Returns: 'sufficient' or 'insufficient'
    """
    try:
        if "QUERY:" not in input_text or "DOCUMENTS:" not in input_text:
            return "insufficient"

        parts = input_text.split("DOCUMENTS:", 1)
        query = parts[0].replace("QUERY:", "").strip()
        documents = parts[1].strip()

        # Quick check - if documents are too short or generic, they're insufficient
        if (not documents or
            documents == "No relevant documents found." or
            len(documents) < 50 or
                "Upcoming Events" in documents):
            return "insufficient"

        # For more thorough evaluation, use LLM
        llm = get_llm()
        prompt = f"""Determine if these documents contain enough information to answer the question.

        Question: {query}
        Documents: {documents}

        Respond with only: "sufficient" or "insufficient"."""

        response = llm.invoke(prompt)
        result = response.content.strip().lower()

        return "sufficient" if "sufficient" in result else "insufficient"

    except Exception as e:
        return "insufficient"


@tool("answer_from_docs_tool")
def answer_from_docs(input_text: str) -> str:
    """Generate answer from documents only"""
    try:
        if "DOCUMENTS:" not in input_text:
            return "Error: Invalid input format"

        parts = input_text.split("DOCUMENTS:", 1)
        query = parts[0].replace("QUERY:", "").strip()
        documents = parts[1].strip()

        llm = get_llm()

        prompt = f"""Answer this question using only the provided documents:

        Question: {query}
        Documents: {documents}

        Provide a comprehensive answer based solely on the document content."""

        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        return f"Error generating answer from documents: {str(e)}"


@tool("web_search_tool")
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo"""
    try:
        search_tool = DuckDuckGoSearchResults(max_results=5)
        results = search_tool.run(query)
        return str(results)  # Ensure it returns a string
    except Exception as e:
        return f"Web search failed: {str(e)}"


@tool("answer_from_web_tool")
def answer_from_web(input_text: str) -> str:
    """Generate answer from web search results"""
    try:
        # Handle case where input might just be the web results without format
        if "RESULTS:" in input_text:
            parts = input_text.split("RESULTS:", 1)
            query = parts[0].replace("QUERY:", "").strip()
            results = parts[1].strip()
        else:
            # If no proper format, treat the entire input as results
            # We need to get the query from session state or default
            query = getattr(st.session_state, 'current_query', 'the question')
            results = input_text

        llm = get_llm()

        prompt = f"""I found that your documents don't contain sufficient information to answer this question, so I searched the web for you. Here's what I found:

        Question: {query}
        Web Search Results: {results}

        Provide a comprehensive answer based on the web search results. Start your response with the above statement about searching the web."""

        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        return f"I searched the web but encountered an error processing the results: {str(e)}"


# Export all tools
all_tools = [evaluate_sufficiency,
             answer_from_docs, web_search, answer_from_web]
