import os
import logging
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger("financial_platform.rag.pdf_loader")

def load_pdf(file_path: str) -> list:
    """
    Load a PDF file and return a list of LangChain Document objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at {file_path}")
        
    logger.info(f"Loading PDF file: {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} pages from {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        raise e
