import argparse
import sys
import shutil
import os
from src.ingestion import DocumentIngestion
from src.indexer import RAGIndexer
from config import Config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_index():
    """
    Initialize or load the RAG index
    
    Returns:
        RAGIndexer object with loaded index
    """
    try:
        # Validate configuration
        Config.validate()
        
        # Initialize indexer
        indexer = RAGIndexer()
        
        # Try to load existing index, create new if needed
        try:
            indexer.create_or_load_index()
        except (ValueError, Exception):
            # No existing index, need to build one
            logger.info("="*60)
            logger.info("No existing index found. Building new index...")
            logger.info("="*60)
            
            # Load and process documents
            ingestion = DocumentIngestion()
            nodes = ingestion.process()
            
            # Create index
            indexer.create_or_load_index(nodes)
            
            logger.info("="*60)
            logger.info("Index built successfully!")
            logger.info("="*60)
        
        return indexer
        
    except Exception as e:
        logger.error(f"Error setting up index: {e}")
        sys.exit(1)


def query(question: str, show_sources: bool = True, verbose: bool = False):
    """
    Process a question and return an answer using RAG
    
    Args:
        question: User's question
        show_sources: Whether to display source documents
        verbose: Show detailed retrieval information
    """
    print("\n" + "="*60)
    print(f"Question: {question}")
    print("="*60 + "\n")
    
    try:
        indexer = setup_index()
        
        query_engine = indexer.get_query_engine()
        
        logger.info("Retrieving relevant passages and generating answer...")
        response = query_engine.query(question)
        
        # Display answer
        print("Answer:")
        print("-" * 60)
        print(response.response)
        print("-" * 60)
        
        # Display sources if available
        if show_sources and hasattr(response, 'source_nodes') and response.source_nodes:
            print("\n Sources:")
            print("-" * 60)
            
            for i, node in enumerate(response.source_nodes, 1):
                filename = node.node.metadata.get('file_name', 'Unknown')
                score = node.score if hasattr(node, 'score') else 'N/A'
                
                print(f"\n[{i}] File: {filename}")
                print(f"    Relevance Score: {score}")
                
                if verbose:
                    text_preview = node.node.text[:200].replace('\n', ' ')
                    print(f"    Preview: {text_preview}...")
            
            print("-" * 60)
        
        print("\n Query completed successfully\n")
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        sys.exit(1)


def rebuild_index():
    """
    Force rebuild of the index from scratch
    """
    try:
        if os.path.exists(Config.STORAGE_DIR):
            logger.info(f"Removing existing index at {Config.STORAGE_DIR}...")
            shutil.rmtree(Config.STORAGE_DIR)
            logger.info("✓ Existing index removed")
        else:
            logger.info("No existing index found")
        
        logger.info("\nIndex will be rebuilt on next query.")
        
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        sys.exit(1)


def main():
    """
    Main entry point for the CLI
    """
    parser = argparse.ArgumentParser(
        description="RAG-powered Q&A Assistant for Internal Documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qa.py --question "How do I reset my password?"
  python qa.py -q "What is the API rate limit?" --verbose
  python qa.py --rebuild
  python qa.py --question "How to authenticate?" --no-sources
        """
    )
    
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Question to ask the assistant"
    )
    
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the index from scratch (clears existing index)"
    )
    
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't display source documents"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information about retrieved passages"
    )
    
    args = parser.parse_args()
    
    # Handle rebuild command
    if args.rebuild:
        rebuild_index()
        return
    
    # Require question if not rebuilding
    if not args.question:
        parser.print_help()
        print("\n Error: --question is required (or use --rebuild)")
        sys.exit(1)
    
    # Process query
    query(
        question=args.question,
        show_sources=not args.no_sources,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()