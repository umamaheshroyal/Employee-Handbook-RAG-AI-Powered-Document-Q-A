import logging
import re

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Perform minimal cleaning on extracted text.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove extra spaces
    text = re.sub(r"[ ]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading and trailing whitespace
    text = text.strip()

    return text


def clean_documents(documents: list[Document]) -> list[Document]:
    """
    Clean the content of LangChain Document objects.
    """

    cleaned_documents = []

    for doc in documents:

        cleaned_doc = Document(
            page_content=clean_text(doc.page_content),
            metadata=doc.metadata
        )

        cleaned_documents.append(cleaned_doc)

    logger.info("Successfully cleaned %d documents.", len(cleaned_documents))

    return cleaned_documents