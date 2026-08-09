import logging
import re

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:

    text = text.replace("\t", " ")

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"[ ]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()

    return text


def clean_documents(documents: list[Document]) -> list[Document]:

    cleaned_documents = []

    for doc in documents:

        cleaned_doc = Document(
            page_content=clean_text(doc.page_content),
            metadata=doc.metadata
        )

        cleaned_documents.append(cleaned_doc)

    logger.info(
        "Successfully cleaned %d documents.",
        len(cleaned_documents)
    )

    return cleaned_documents