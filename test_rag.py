import os
import sys
from langchain_core.documents import Document

# Ensure backend directory is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from app.rag.vector_store import add_documents_to_store, query_vector_store
from app.agents.rag_agent import answer_query_with_context, run_rag_agent

def main():
    print("--- Testing Corporate Report RAG System ---")
    ticker = "TCS"
    
    # 1. Create mock chunks for TCS
    chunks = [
        Document(
            page_content="Tata Consultancy Services (TCS) reported a net profit of 12,040 Crore INR for Q4 FY26, representing a growth of 9.1% year-on-year. The revenue reached 61,230 Crore INR driven by cloud migration and digital transformation contracts.",
            metadata={"source": "annual_report_tcs.pdf", "page": 0}
        ),
        Document(
            page_content="Key risks for TCS include high talent turnover in cognitive computing divisions, rising supply-side costs in Western Europe, and geopolitical tensions impacting client spending in BFSI sector.",
            metadata={"source": "annual_report_tcs.pdf", "page": 4}
        )
    ]
    
    print(f"Indexing {len(chunks)} mock documents for ticker {ticker}...")
    add_documents_to_store(chunks, ticker)
    
    # 2. Query vector store directly
    query = "What is the net profit of TCS?"
    print(f"\nQuerying vector store for: '{query}'...")
    results = query_vector_store(query, ticker, k=1)
    if results:
        print("✔ Found match:")
        print(f"  Source: {results[0].metadata.get('source')}, Page: {results[0].metadata.get('page')}")
        print(f"  Content: {results[0].page_content}")
    else:
        print("❌ No matches found!")
        
    # 3. Query RAG agent (Gemini QA)
    print(f"\nTriggering Gemini RAG Chat for query: '{query}'...")
    try:
        response = answer_query_with_context(query, ticker)
        print("✔ Gemini Advisory Response:")
        print(response.get("answer"))
        print("\nCitations:")
        print(response.get("sources"))
    except Exception as e:
        print(f"❌ Gemini QA failed: {e}")
        
    # 4. Test RAG LangGraph Agent Node
    print("\nTesting RAG Node in Supervisor state...")
    state = {
        "ticker": ticker,
        "history_data": None
    }
    node_output = run_rag_agent(state)
    print("✔ RAG Node Output:")
    print(node_output.get("rag_data", {}).get("summary")[:300] + "...")

if __name__ == "__main__":
    main()
