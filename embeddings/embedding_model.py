import logging

from langchain_openai import OpenAIEmbeddings

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


def get_embedding_model():
    """
    Create and return the OpenAI embedding model.
    """

    logger.info("Loading embedding model...")

    embedding_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    logger.info("Embedding model loaded successfully.")

    return embedding_model