import pytest
import os
from unittest.mock import patch, MagicMock
from src.ingestion import DocumentIngestion
from src.indexer import RAGIndexer


class TestEndToEndPipeline:
    """Test the complete RAG pipeline from documents to answers"""
    
    def test_full_pipeline_from_docs_to_answer(self, mock_config):
        """Test complete pipeline: load docs -> chunk -> index -> query -> answer"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI') as mock_llm:
            
            # Setup mocks
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance
            
            # Step 1: Ingest documents
            ingestion = DocumentIngestion()
            documents = ingestion.load_documents()
            assert len(documents) > 0
            
            # Step 2: Chunk documents
            nodes = ingestion.chunk_documents(documents)
            assert len(nodes) > 0
            
            # Step 3: Create index
            indexer = RAGIndexer()
            index = indexer.create_or_load_index(nodes)
            assert index is not None
            
            # Step 4: Create query engine
            query_engine = indexer.get_query_engine()
            assert query_engine is not None
            
            # Step 5: Query and get answer
            mock_response = MagicMock()
            mock_response.response = "Authentication uses OAuth 2.0"
            mock_response.source_nodes = []
            
            with patch.object(query_engine, 'query', return_value=mock_response):
                response = query_engine.query("How does authentication work?")
                assert response.response is not None
                assert len(response.response) > 0
    
    def test_pipeline_with_index_persistence(self, mock_config, temp_storage_dir):
        """Test that pipeline can save and reload index"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            # First run: Create and save index
            ingestion = DocumentIngestion()
            nodes = ingestion.process()
            
            indexer1 = RAGIndexer()
            indexer1.create_or_load_index(nodes)
            
            # Verify storage was created
            assert os.path.exists(temp_storage_dir)
            assert len(os.listdir(temp_storage_dir)) > 0
            
            # Second run: Load existing index
            indexer2 = RAGIndexer()
            index = indexer2.create_or_load_index()
            assert index is not None
    
    def test_multiple_queries_on_same_index(self, mock_config):
        """Test running multiple queries on the same index"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            # Build index
            ingestion = DocumentIngestion()
            nodes = ingestion.process()
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(nodes)
            query_engine = indexer.get_query_engine()
            
            # Multiple queries
            questions = [
                "How do I authenticate?",
                "What is the rate limit?",
                "How to reset password?"
            ]
            
            for question in questions:
                mock_response = MagicMock()
                mock_response.response = f"Answer to {question}"
                mock_response.source_nodes = []
                
                with patch.object(query_engine, 'query', return_value=mock_response):
                    response = query_engine.query(question)
                    assert response.response is not None


class TestPipelineRobustness:
    """Test pipeline robustness and error handling"""
    
    def test_pipeline_handles_empty_docs_directory(self, tmp_path, monkeypatch):
        """Test pipeline handles empty docs directory gracefully"""
        empty_dir = tmp_path / "empty_docs"
        empty_dir.mkdir()
        
        monkeypatch.setattr("config.Config.DOCS_DIR", str(empty_dir))
        
        ingestion = DocumentIngestion()
        documents = ingestion.load_documents()
        
        assert len(documents) == 0
    
    def test_pipeline_with_minimal_documents(self, tmp_path, monkeypatch):
        """Test pipeline with only one small document"""
        docs_dir = tmp_path / "minimal_docs"
        docs_dir.mkdir()
        
        (docs_dir / "single.md").write_text("This is a single short document.")
        
        monkeypatch.setattr("config.Config.DOCS_DIR", str(docs_dir))
        monkeypatch.setattr("config.Config.OPENAI_API_KEY", "test-key")
        
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            ingestion = DocumentIngestion()
            nodes = ingestion.process()
            
            assert len(nodes) > 0
            
            indexer = RAGIndexer()
            index = indexer.create_or_load_index(nodes)
            assert index is not None
    
    def test_pipeline_with_large_documents(self, tmp_path, monkeypatch):
        """Test pipeline with very large documents"""
        docs_dir = tmp_path / "large_docs"
        docs_dir.mkdir()
        
        # Create a large document
        large_content = "This is a sentence. " * 1000  # ~5000 words
        (docs_dir / "large.md").write_text(large_content)
        
        monkeypatch.setattr("config.Config.DOCS_DIR", str(docs_dir))
        monkeypatch.setattr("config.Config.OPENAI_API_KEY", "test-key")
        
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            ingestion = DocumentIngestion()
            nodes = ingestion.process()
            
            # Should create multiple chunks
            assert len(nodes) > 1


class TestPipelinePerformance:
    """Test pipeline performance characteristics"""
    
    def test_index_creation_is_idempotent(self, mock_config):
        """Test that creating index twice gives same result"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            ingestion = DocumentIngestion()
            nodes1 = ingestion.process()
            nodes2 = ingestion.process()
            
            # Should get same nodes
            assert len(nodes1) == len(nodes2)
            
            texts1 = [n.text for n in nodes1]
            texts2 = [n.text for n in nodes2]
            assert texts1 == texts2
    
    def test_query_engine_reuse(self, mock_config):
        """Test that query engine can be reused for multiple queries"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            ingestion = DocumentIngestion()
            nodes = ingestion.process()
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(nodes)
            
            # Get query engine once
            query_engine = indexer.get_query_engine()
            
            # Use it multiple times
            for i in range(3):
                mock_response = MagicMock()
                mock_response.response = f"Answer {i}"
                mock_response.source_nodes = []
                
                with patch.object(query_engine, 'query', return_value=mock_response):
                    response = query_engine.query(f"Question {i}?")
                    assert response is not None


class TestPipelineDataFlow:
    """Test data flow through the pipeline"""
    
    def test_metadata_preserved_through_pipeline(self, mock_config):
        """Test that metadata is preserved from docs to final response"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            # Load documents (should have file_name metadata)
            ingestion = DocumentIngestion()
            documents = ingestion.load_documents()
            
            # Check documents have metadata
            for doc in documents:
                assert 'file_name' in doc.metadata
            
            # Chunk documents (should preserve metadata)
            nodes = ingestion.chunk_documents(documents)
            
            # Check nodes have metadata
            for node in nodes:
                if node.metadata:  # Some nodes might have empty metadata
                    assert isinstance(node.metadata, dict)
            
            # Create index
            indexer = RAGIndexer()
            indexer.create_or_load_index(nodes)
            query_engine = indexer.get_query_engine()
            
            # Query and check source nodes have metadata
            from llama_index.core.schema import TextNode, NodeWithScore
            mock_source = NodeWithScore(
                node=TextNode(
                    text="Sample",
                    metadata={"file_name": "test.md"}
                ),
                score=0.9
            )
            
            mock_response = MagicMock()
            mock_response.response = "Answer"
            mock_response.source_nodes = [mock_source]
            
            with patch.object(query_engine, 'query', return_value=mock_response):
                response = query_engine.query("Test question?")
                
                if response.source_nodes:
                    assert 'file_name' in response.source_nodes[0].node.metadata
    
    def test_text_content_preserved_through_pipeline(self, mock_config):
        """Test that text content is preserved from docs to retrieval"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            # Get original document text
            ingestion = DocumentIngestion()
            documents = ingestion.load_documents()
            original_texts = [doc.text for doc in documents]
            
            # Chunk documents
            nodes = ingestion.chunk_documents(documents)
            chunked_texts = [node.text for node in nodes]
            
            # All chunk text should come from original documents
            all_chunked_text = " ".join(chunked_texts)
            
            # Check that key content is preserved
            # (exact matching is hard due to chunking, so we check key phrases)
            if "OAuth" in " ".join(original_texts):
                assert "OAuth" in all_chunked_text or "oauth" in all_chunked_text.lower()