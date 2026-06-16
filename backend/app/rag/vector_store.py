import logging
from langchain_community.vectorstores import Chroma
from app.config import settings
from app.rag.embedder import get_embedding_model

logger = logging.getLogger("financial_platform.rag.vector_store")

def get_vector_store(collection_name: str = "corporate_reports"):
    """
    Get or create the Chroma vector store instance.
    """
    embeddings = get_embedding_model()
    db_dir = settings.CHROMA_DB_DIR
    
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=db_dir
    )

def add_documents_to_store(chunks: list, ticker: str, collection_name: str = "corporate_reports"):
    """
    Tag all document chunks with the ticker metadata, then store them in ChromaDB.
    """
    ticker_clean = ticker.upper().strip()
    logger.info(f"Adding {len(chunks)} chunks to vector store for ticker: {ticker_clean}")
    
    for chunk in chunks:
        if not chunk.metadata:
            chunk.metadata = {}
        chunk.metadata["ticker"] = ticker_clean
        
    vector_store = get_vector_store(collection_name)
    vector_store.add_documents(chunks)
    logger.info(f"Successfully added chunks to vector store for {ticker_clean}.")

def query_vector_store(query: str, ticker: str, k: int = 4, collection_name: str = "corporate_reports") -> list:
    """
    Query the vector store for a search term, filtering specifically by stock ticker.
    """
    ticker_clean = ticker.upper().strip()
    logger.info(f"Querying vector store for '{query}' | ticker: {ticker_clean} (k={k})")
    vector_store = get_vector_store(collection_name)
    
    # Query with metadata filter
    results = vector_store.similarity_search(
        query,
        k=k,
        filter={"ticker": ticker_clean}
    )
    logger.info(f"Retrieved {len(results)} chunks from vector store.")
    return results
