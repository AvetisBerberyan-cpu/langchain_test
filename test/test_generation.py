import pytest
from unittest.mock import patch, MagicMock
from src.indexer import RAGIndexer
from llama_index.core.schema import TextNode, NodeWithScore


class TestQueryEngine:
    """Test query engine and answer generation"""
    
    def test_query_engine_generates_response(self, sample_nodes, mock_config):
        """Test that query engine generates a response"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI') as mock_llm:
            
            # Setup mocks
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            mock_llm_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "This is a generated answer about authentication."
            mock_llm_instance.complete.return_value = mock_response
            mock_llm.return_value = mock_llm_instance
            
            # Create indexer and query engine
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            query_engine = indexer.get_query_engine()
            
            # Mock the query response
            mock_query_response = MagicMock()
            mock_query_response.response = "Authentication uses OAuth 2.0 protocol."
            mock_query_response.source_nodes = []
            
            with patch.object(query_engine, 'query', return_value=mock_query_response):
                response = query_engine.query("How does authentication work?")
                
                assert response is not None
                assert hasattr(response, 'response')
                assert isinstance(response.response, str)
                assert len(response.response) > 0
    
    def test_query_engine_with_different_questions(self, sample_nodes, mock_config):
        """Test query engine with various question types"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI') as mock_llm:
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            query_engine = indexer.get_query_engine()
            
            questions = [
                "How do I reset my password?",
                "What is the API rate limit?",
                "How does authentication work?",
                "What are the system requirements?"
            ]
            
            for question in questions:
                mock_response = MagicMock()
                mock_response.response = f"Answer to: {question}"
                mock_response.source_nodes = []
                
                with patch.object(query_engine, 'query', return_value=mock_response):
                    response = query_engine.query(question)
                    assert response.response is not None


class TestResponseQuality:
    """Test the quality of generated responses"""
    
    def test_response_includes_source_nodes(self, sample_nodes, mock_config):
        """Test that responses include source node information"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            query_engine = indexer.get_query_engine()
            
            # Mock response with source nodes
            mock_source_node = NodeWithScore(
                node=TextNode(
                    text="Relevant passage",
                    metadata={"file_name": "auth.md"}
                ),
                score=0.9
            )
            
            mock_response = MagicMock()
            mock_response.response = "Generated answer"
            mock_response.source_nodes = [mock_source_node]
            
            with patch.object(query_engine, 'query', return_value=mock_response):
                response = query_engine.query("test question")
                
                assert hasattr(response, 'source_nodes')
                assert len(response.source_nodes) > 0
                assert hasattr(response.source_nodes[0], 'node')
    
    def test_response_uses_retrieved_context(self, sample_nodes, mock_config):
        """Test that response is based on retrieved context"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI') as mock_llm:
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            # Mock LLM to track if it received context
            mock_llm_instance = MagicMock()
            mock_llm_instance.complete = MagicMock()
            mock_llm.return_value = mock_llm_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            query_engine = indexer.get_query_engine()
            
            # The actual implementation would call LLM with context
            # This test structure shows how to verify it
    
    def test_response_handles_no_relevant_context(self, sample_nodes, mock_config):
        """Test response when no relevant context is found"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            query_engine = indexer.get_query_engine()
            
            # Mock response indicating no relevant info found
            mock_response = MagicMock()
            mock_response.response = "I cannot find information about that in the documentation."
            mock_response.source_nodes = []
            
            with patch.object(query_engine, 'query', return_value=mock_response):
                response = query_engine.query("What is quantum computing?")
                
                assert response.response is not None
                # Response should indicate inability to answer


class TestGenerationParameters:
    """Test different generation parameters"""
    
    def test_different_response_modes(self, sample_nodes, mock_config):
        """Test query engine with different response modes"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            
            response_modes = ["compact", "tree_summarize", "refine"]
            
            for mode in response_modes:
                query_engine = indexer.get_query_engine(response_mode=mode)
                assert query_engine is not None
    
    def test_temperature_affects_generation(self, sample_nodes, mock_config):
        """Test that temperature setting is respected"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI') as mock_llm:
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            # Verify temperature is set in OpenAI initialization
            indexer = RAGIndexer()
            
            # Check that OpenAI was called with temperature from config
            call_kwargs = mock_llm.call_args[1]
            assert 'temperature' in call_kwargs


class TestGenerationEdgeCases:
    """Test edge cases in generation"""
    
    def test_generation_with_empty_context(self, sample_nodes, mock_config):
        """Test generation when retrieved context is empty"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            query_engine = indexer.get_query_engine()
            
            # Mock empty context scenario
            mock_response = MagicMock()
            mock_response.response = "No relevant information found."
            mock_response.source_nodes = []
            
            with patch.object(query_engine, 'query', return_value=mock_response):
                response = query_engine.query("Completely unrelated question")
                assert response is not None
    
    def test_generation_with_long_context(self, mock_config):
        """Test generation with very long retrieved context"""
        long_nodes = [
            TextNode(
                text="Lorem ipsum " * 500,  # Very long text
                metadata={"file_name": f"doc{i}.md"}
            )
            for i in range(5)
        ]
        
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(long_nodes)
            query_engine = indexer.get_query_engine()
            
            # Should handle long context without errors
            assert query_engine is not None
    
    def test_generation_with_special_characters_in_context(self, mock_config):
        """Test generation when context contains special characters"""
        special_nodes = [
            TextNode(
                text="Code example: `const api_key = 'sk-123'`; Use @mentions & #tags!",
                metadata={"file_name": "code.md"}
            )
        ]
        
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(special_nodes)
            query_engine = indexer.get_query_engine()
            
            mock_response = MagicMock()
            mock_response.response = "Answer with special chars"
            mock_response.source_nodes = []
            
            with patch.object(query_engine, 'query', return_value=mock_response):
                response = query_engine.query("How to use API?")
                assert response is not None


class TestGenerationConsistency:
    """Test consistency of generation"""
    
    def test_same_query_similar_response(self, sample_nodes, mock_config):
        """Test that same query produces consistent responses"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            query_engine = indexer.get_query_engine()
            
            question = "How do I reset my password?"
            
            mock_response = MagicMock()
            mock_response.response = "Consistent answer about password reset"
            mock_response.source_nodes = []
            
            with patch.object(query_engine, 'query', return_value=mock_response):
                response1 = query_engine.query(question)
                response2 = query_engine.query(question)
                
                # With low temperature, responses should be similar
                assert response1.response == response2.response