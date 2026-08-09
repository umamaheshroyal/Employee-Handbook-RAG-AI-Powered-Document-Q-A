import logging

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

logger = logging.getLogger(__name__)


def create_rag_chain(
    retriever,
    prompt,
    llm,
):
    """
    Create the Retrieval-Augmented Generation (RAG) chain.

    Args:
        retriever: LangChain retriever.
        prompt: ChatPromptTemplate.
        llm: ChatOpenAI model.

    Returns:
        RetrievalChain
    """

    logger.info("Creating document chain...")

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    logger.info("Creating retrieval chain...")

    rag_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    logger.info("RAG chain created successfully.")

    return rag_chain
