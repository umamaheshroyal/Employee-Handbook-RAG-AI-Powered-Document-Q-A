import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from loaders import load_pdf
from preprocessing.cleaner import clean_documents
from chunking.text_splitter import split_documents
from embeddings.embedding_model import get_embedding_model
from vectorstore.faiss_store import create_vector_store
from retrieval.retriever import create_retriever
from prompts.prompt_template import get_rag_prompt
from llm.llm import get_llm
from pipeline.rag_pipeline import create_rag_chain


# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="Employee Handbook RAG API",
    version="1.0.0"
)


# --------------------------------------------------
# RAG Chain
# --------------------------------------------------

rag_chain = None


# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


# --------------------------------------------------
# Startup Event
# --------------------------------------------------

@app.on_event("startup")
def startup():

    global rag_chain

    logger.info("Loading RAG Pipeline...")

    # 1. Load PDF
    documents = load_pdf(
        r"C:\Users\ASUS\Downloads\Employee-Handbook.pdf"
    )

    # 2. Clean documents
    cleaned_documents = clean_documents(documents)

    # 3. Split documents into chunks
    chunks = split_documents(cleaned_documents)

    # 4. Load embedding model
    embedding_model = get_embedding_model()

    # 5. Create FAISS vector store
    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    # 6. Create retriever
    retriever = create_retriever(vector_store)

    # 7. Create RAG prompt
    prompt = get_rag_prompt()

    # 8. Load LLM
    llm = get_llm()

    # 9. Create RAG chain
    rag_chain = create_rag_chain(
        retriever,
        prompt,
        llm
    )

    logger.info("RAG API Ready")


# --------------------------------------------------
# Home Endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Employee Handbook RAG API is Running"
    }


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# RAG Query Endpoint
# --------------------------------------------------

@app.post("/query")
def query(request: QuestionRequest):

    try:

        if rag_chain is None:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline is not initialized"
            )

        response = rag_chain.invoke(
            {
                "input": request.question
            }
        )

        return {
            "question": request.question,
            "answer": response["answer"]
        }

    except HTTPException:
        raise

    except Exception as e:

        logger.exception("Error while processing query")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )