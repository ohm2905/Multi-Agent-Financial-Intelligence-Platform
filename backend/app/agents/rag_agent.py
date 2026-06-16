import os
import logging
import google.generativeai as genai

from app.config import settings
from app.rag.vector_store import get_vector_store, query_vector_store

logger = logging.getLogger("financial_platform.agents.rag")

def get_gemini_model():
    """Configure and return the Gemini 2.5 Flash model."""
    if not settings.GEMINI_API_KEY or "your_gemini_api_key" in settings.GEMINI_API_KEY:
        raise ValueError("Gemini API key is not configured. Please set GEMINI_API_KEY in your .env file.")
    genai.configure(api_key=settings.GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.5-flash")

def has_corporate_reports(ticker: str) -> bool:
    """Check if ChromaDB has any document chunks indexed for this ticker."""
    ticker_clean = ticker.upper().strip()
    try:
        vector_store = get_vector_store()
        res = vector_store.get(where={"ticker": ticker_clean}, limit=1)
        return len(res.get("ids", [])) > 0
    except Exception as e:
        logger.error(f"Error checking corporate reports in ChromaDB: {e}")
        return False

def answer_query_with_context(query: str, ticker: str) -> dict:
    """
    Retrieve matching context chunks from ChromaDB and formulate an answer using Gemini 2.5 Flash.
    """
    ticker_clean = ticker.upper().strip()
    if not has_corporate_reports(ticker_clean):
        return {
            "answer": f"No corporate reports have been uploaded or indexed for {ticker_clean} yet. Please upload a PDF annual report/filing to ask questions.",
            "sources": []
        }
        
    try:
        # 1. Retrieve similar chunks
        chunks = query_vector_store(query, ticker_clean, k=5)
        
        # 2. Compile context text and keep track of pages/sources
        context_parts = []
        sources = []
        for idx, doc in enumerate(chunks):
            page = doc.metadata.get("page", 0) + 1
            source_file = os.path.basename(doc.metadata.get("source", "Report"))
            context_parts.append(f"--- Context Chunk {idx+1} (Source: {source_file}, Page: {page}) ---\n{doc.page_content}")
            
            # De-duplicate sources list
            src_info = {"source": source_file, "page": page}
            if src_info not in sources:
                sources.append(src_info)
            
        context_text = "\n\n".join(context_parts)
        
        # 3. Formulate Prompt for Gemini
        prompt = (
            f"You are an expert financial analyst. Answer the user's question about {ticker_clean} "
            f"using the provided context chunks extracted from the company's official corporate reports.\n\n"
            f"Context:\n{context_text}\n\n"
            f"User Question: {query}\n\n"
            f"Guidelines:\n"
            f"- Base your answer strictly on the provided context. If the information is not in the context, state that clearly.\n"
            f"- Be professional, quantitative, and objective.\n"
            f"- Cite page numbers and source files from the context when reporting facts (e.g. 'on Page 15 of Annual Report...').\n"
            f"- Do not speculate or make up information.\n\n"
            f"Advisory Response:"
        )
        
        # 4. Generate Content
        model = get_gemini_model()
        response = model.generate_content(prompt)
        
        return {
            "answer": response.text.strip(),
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"Error answering query with RAG context: {e}")
        return {
            "answer": f"An error occurred while formulating response: {str(e)}",
            "sources": []
        }

def run_rag_agent(state: dict) -> dict:
    """
    RAG Agent runner node for LangGraph workflow.
    Performs an automatic summary analysis of the corporate reports if available.
    """
    ticker = state.get("ticker")
    try:
        logger.info(f"Running RAG Agent for ticker: {ticker}")
        
        if not has_corporate_reports(ticker):
            return {
                "rag_data": {
                    "summary": "No corporate reports uploaded yet for this ticker. Upload an annual report or quarterly filing PDF to enable AI report synthesis.",
                    "available": False,
                    "sources": []
                }
            }
            
        # If documents exist, perform a standard strategic audit query to include in report
        audit_query = "Summarize the key highlights, financial performance, strategic goals, and risk factors of the company."
        logger.info(f"RAG Agent performing default report summary for {ticker}...")
        res = answer_query_with_context(audit_query, ticker)
        
        return {
            "rag_data": {
                "summary": res["answer"],
                "available": True,
                "sources": res["sources"]
            }
        }
    except Exception as e:
        logger.error(f"Error in RAG Agent execution: {e}")
        return {
            "rag_data": {
                "summary": f"RAG Agent execution failed: {str(e)}",
                "available": False,
                "sources": [],
                "error": str(e)
            }
        }
