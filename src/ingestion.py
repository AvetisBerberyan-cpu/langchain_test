from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIngestion:
    """
    Handles document loading and chunking for the RAG pipeline
    """
    
    def __init__(self, chunk_size=None, chunk_overlap=None):
        """
        Initialize the document ingestion system
        
        Args:
            chunk_size: Size of each chunk in tokens (default from Config)
            chunk_overlap: Overlap between chunks (default from Config)
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.docs_dir = Config.DOCS_DIR
        
        logger.info(f"Initialized DocumentIngestion with chunk_size={self.chunk_size}, "
                   f"overlap={self.chunk_overlap}")
    
    def load_documents(self):
        """
        Load all markdown and text files from the documents directory
        
        Returns:
            List of Document objects containing text and metadata
        """
        try:
            logger.info(f"Loading documents from {self.docs_dir}...")
            
            reader = SimpleDirectoryReader(
                input_dir=self.docs_dir,
                recursive=True,
                required_exts=[".md", ".txt"],
                filename_as_id=True
            )
            
            documents = reader.load_data()
            
            logger.info(f"✓ Successfully loaded {len(documents)} documents")
            
            # Log document details
            for i, doc in enumerate(documents, 1):
                filename = doc.metadata.get('file_name', 'Unknown')
                char_count = len(doc.text)
                logger.debug(f"  [{i}] {filename} ({char_count} characters)")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def chunk_documents(self, documents):
        """
        Split documents into smaller chunks for better retrieval
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Node objects (chunks) with metadata
        """
        try:
            logger.info("Chunking documents...")
            
            # Create sentence splitter
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                paragraph_separator="\n\n",
                secondary_chunking_regex="[.!?]\\s+"
            )
            
            # Convert documents to nodes (chunks)
            nodes = splitter.get_nodes_from_documents(documents)
            
            logger.info(f"✓ Created {len(nodes)} chunks from {len(documents)} documents")
            
            # Log chunking statistics
            avg_chunk_size = sum(len(node.text) for node in nodes) / len(nodes)
            logger.debug(f"  Average chunk size: {avg_chunk_size:.0f} characters")
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise
    
    def process(self):
        """
        Complete ingestion pipeline: load and chunk documents
        
        Returns:
            List of Node objects ready for indexing
        """
        documents = self.load_documents()
        nodes = self.chunk_documents(documents)
        return nodes