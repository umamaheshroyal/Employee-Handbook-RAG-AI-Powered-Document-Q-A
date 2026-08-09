import logging

from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)


def create_retriever(
    vector_store: FAISS,
    search_type: str = "similarity",
    k: int = 4
) -> VectorStoreRetriever:
    """
    Create a retriever from a FAISS vector store.

    Args:
        vector_store: FAISS vector store.
        search_type: similarity, mmr, or similarity_score_threshold.
        k: Number of chunks to retrieve.

    Returns:
        VectorStoreRetriever
    """

    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": k
        }
    )

    logger.info("Retriever created successfully.")

    return retriever