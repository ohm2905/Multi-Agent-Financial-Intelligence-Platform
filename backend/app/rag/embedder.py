import logging
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger("financial_platform.rag.embedder")

_embeddings = None

def get_embedding_model():
    """
    Get or initialize the HuggingFaceEmbeddings model (BAAI/bge-small-en-v1.5).
    Uses HF_HOME environment variable to cache weights on external storage.
    """
    global _embeddings
    if _embeddings is not None:
        return _embeddings
        
    logger.info("Initializing RAG embedding model (BAAI/bge-small-en-v1.5)...")
    try:
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"}  # CPUs handle BGE Small efficiently
        )
        logger.info("RAG embedding model initialized successfully.")
        return _embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        raise e
