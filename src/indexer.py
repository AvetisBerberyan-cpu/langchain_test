from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from config import Config
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGIndexer:
    """
    Handles vector index creation, storage, and querying
    """
    
    def __init__(self):
        """
        Initialize the RAG indexer with embedding and LLM models
        """
        logger.info("Initializing RAG Indexer...")
        
        # Configure embedding model
        logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Configure LLM
        logger.info(f"Configuring LLM: {Config.LLM_MODEL}")
        Settings.llm = OpenAI(
            model=Config.LLM_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=Config.LLM_TEMPERATURE
        )
        
        self.index = None
        logger.info("✓ RAG Indexer initialized successfully")
    
    def create_or_load_index(self, nodes=None):
        """
        Create a new index or load existing one from storage
        
        Args:
            nodes: List of Node objects to index (required if creating new)
            
        Returns:
            VectorStoreIndex object
        """
        if os.path.exists(Config.STORAGE_DIR):
            try:
                logger.info(f"Loading existing index from {Config.STORAGE_DIR}...")
                storage_context = StorageContext.from_defaults(
                    persist_dir=Config.STORAGE_DIR
                )
                self.index = load_index_from_storage(storage_context)
                logger.info("✓ Successfully loaded existing index")
                return self.index
            except Exception as e:
                logger.warning(f"Could not load existing index: {e}")
                logger.info("Will create new index...")
        
        # Create new index
        if nodes is None:
            raise ValueError(
                "No existing index found and no nodes provided. "
                "Please provide nodes to create a new index."
            )
        
        logger.info(f"Creating new vector index from {len(nodes)} chunks...")
        logger.info("This may take a few minutes for embedding generation...")
        
        try:
            self.index = VectorStoreIndex(nodes, show_progress=True)
            
            logger.info(f"Saving index to {Config.STORAGE_DIR}...")
            self.index.storage_context.persist(persist_dir=Config.STORAGE_DIR)
            
            logger.info("✓ Successfully created and saved new index")
            return self.index
            
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise
    
    def get_query_engine(self, similarity_top_k=None, response_mode=None):
        """
        Get a query engine for retrieval and generation
        
        Args:
            similarity_top_k: Number of chunks to retrieve (default from Config)
            response_mode: How to synthesize response (default from Config)
            
        Returns:
            Query engine configured for RAG
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_or_load_index() first.")
        
        top_k = similarity_top_k or Config.TOP_K
        mode = response_mode or Config.RESPONSE_MODE
        
        logger.info(f"Creating query engine (top_k={top_k}, mode={mode})...")
        
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode=mode,
            streaming=False
        )
        
        return query_engine
    
    def get_retriever(self, similarity_top_k=None):
        """
        Get a retriever for getting relevant chunks without generation
        
        Args:
            similarity_top_k: Number of chunks to retrieve
            
        Returns:
            Retriever object
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_or_load_index() first.")
        
        top_k = similarity_top_k or Config.TOP_K
        
        return self.index.as_retriever(similarity_top_k=top_k)