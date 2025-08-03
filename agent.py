# agent.py - ROBUST CUSTOM CHAIN
import streamlit as st
from tools import evaluate_sufficiency, answer_from_docs, web_search, answer_from_web


class DocumentWebSearchChain:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, inputs):
        """Custom chain with reliable error handling"""
        query = inputs["input"]
        retrieved_docs = inputs["retrieved_documents"]

        # Store current query for tools that might need it
        st.session_state.current_query = query

        steps = []

        try:
            # Step 1: Evaluate sufficiency
            eval_input = f"QUERY: {query} DOCUMENTS: {retrieved_docs}"
            sufficiency_result = evaluate_sufficiency.func(eval_input)
            steps.append(("evaluate_sufficiency_tool",
                         eval_input, sufficiency_result))

            if sufficiency_result == "sufficient":
                # Step 2a: Answer from documents
                try:
                    answer = answer_from_docs.func(eval_input)
                    steps.append(("answer_from_docs_tool", eval_input, answer))
                    final_answer = answer
                except Exception as e:
                    final_answer = f"Error generating answer from documents: {str(e)}"
            else:
                # Step 2b: Web search and answer
                try:
                    web_results = web_search.func(query)
                    steps.append(("web_search_tool", query, web_results))

                    # Step 3: Answer from web results
                    web_input = f"QUERY: {query} RESULTS: {web_results}"
                    web_answer = answer_from_web.func(web_input)
                    steps.append(
                        ("answer_from_web_tool", web_input, web_answer))
                    final_answer = web_answer
                except Exception as e:
                    final_answer = f"I found that your documents don't contain sufficient information to answer this question. I attempted to search the web but encountered an error: {str(e)}"

            return {
                "output": final_answer,
                "intermediate_steps": [(step[0], step[1][:100] + "...", step[2][:200] + "...") for step in steps]
            }

        except Exception as e:
            return {
                "output": f"Error in processing: {str(e)}",
                "intermediate_steps": steps
            }


def get_agent_executor(llm):
    """Return custom chain"""
    return DocumentWebSearchChain(llm)
