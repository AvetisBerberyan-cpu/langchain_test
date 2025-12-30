import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Provides configuration parameters for the RAG model.
    """

    DOCS_DIR = "./docs"
    STORAGE_DIR = "./storage"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TOP_K = 3
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-3.5-turbo"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
