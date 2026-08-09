import logging

from loaders import load_pdf
from preprocessing.cleaner import clean_documents
from chunking.text_splitter import split_documents
from embeddings.embedding_model import get_embedding_model
from vectorstore.faiss_store import create_vector_store
from retrieval.retriever import create_retriever
from prompts.prompt_template import get_rag_prompt
from llm.llm import get_llm
from pipeline.rag_pipeline import create_rag_chain



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():

    logger.info("Starting RAG Pipeline")

    # 1. Load PDF
    documents = load_pdf(
        r"C:\Users\ASUS\Downloads\Employee-Handbook.pdf"
    )

    # 2. Clean documents
    cleaned_documents = clean_documents(documents)

    # 3. Create chunks
    chunks = split_documents(cleaned_documents)

    # 4. Create embedding model
    embedding_model = get_embedding_model()

    # 5. Create vector store
    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    # 6. Create retriever
    retriever = create_retriever(vector_store)

    # 7. Prompt
    prompt = get_rag_prompt()

    # 8. LLM
    llm = get_llm()

    # 9. RAG chain
    rag_chain = create_rag_chain(
        retriever,
        prompt,
        llm
    )

    logger.info("RAG Bot is ready!")

    while True:

        question = input("\nAsk a Question (type 'exit' to quit): ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        try:

            response = rag_chain.invoke(
                {
                    "input": question
                }
            )

            print("\nAnswer:\n")
            print(response["answer"])

        except Exception as e:

            logger.exception(e)
            print(f"\nError: {e}")


if __name__ == "__main__":
    main() 