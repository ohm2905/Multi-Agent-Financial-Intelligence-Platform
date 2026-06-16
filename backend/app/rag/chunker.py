import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("financial_platform.rag.chunker")

def split_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Split a list of documents into overlapping chunks.
    """
    logger.info(f"Splitting {len(documents)} pages into chunks (size: {chunk_size}, overlap: {chunk_overlap})")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Successfully split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise e
