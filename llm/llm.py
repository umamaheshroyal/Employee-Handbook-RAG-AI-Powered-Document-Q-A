import logging

from langchain_openai import ChatOpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    LLM_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
)

logger = logging.getLogger(__name__)


def get_llm():
    """
    Load and return the LLM.
    """

    logger.info("Loading LLM...")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        max_tokens=MAX_TOKENS,
    )

    logger.info("LLM loaded successfully.")

    return llm