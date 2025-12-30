import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Central configuration for the RAG system"""
    
    # Directory paths
    DOCS_DIR = "./docs"
    STORAGE_DIR = "./storage"
    
    # Document processing settings
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    
    # Retrieval settings
    TOP_K = 3
    
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-3.5-turbo"
    # LLM_MODEL = "llama-3.1-8b-instant"
    
    # API key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Optional: Advanced settings
    LLM_TEMPERATURE = 0.1
    RESPONSE_MODE = "compact"
    
    @classmethod
    def validate(cls):
        """Validate that required configurations are set"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Please set it in .env file or environment variables."
            )
        
        if not os.path.exists(cls.DOCS_DIR):
            raise ValueError(
                f"Documents directory '{cls.DOCS_DIR}' not found. "
                "Please create it and add your documentation files."
            )