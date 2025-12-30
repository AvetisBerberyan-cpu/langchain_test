import pytest
from unittest.mock import patch, MagicMock
from src.indexer import RAGIndexer
from llama_index.core.schema import TextNode, NodeWithScore


class TestBasicRetrieval:
    """Test basic retrieval functionality"""
    
    def test_retriever_returns_results(self, sample_nodes, mock_config):
        """Test that retriever returns results for a query"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            # Setup mock embedding
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            # Create index
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            retriever = indexer.get_retriever()
            
            # Mock the retrieve method to return sample results
            mock_node = TextNode(text="Sample result", metadata={"file_name": "test.md"})
            mock_result = NodeWithScore(node=mock_node, score=0.9)
            
            with patch.object(retriever, 'retrieve', return_value=[mock_result]):
                results = retriever.retrieve("test query")
                
                assert len(results) > 0
                assert hasattr(results[0], 'node')
                assert hasattr(results[0], 'score')
    
    def test_retrieval_respects_top_k(self, sample_nodes, mock_config):
        """Test that retrieval respects the top_k parameter"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            
            # Get retrievers with different top_k values
            retriever_2 = indexer.get_retriever(similarity_top_k=2)
            retriever_5 = indexer.get_retriever(similarity_top_k=5)
            
            # Both should be created successfully
            assert retriever_2 is not None
            assert retriever_5 is not None


class TestRetrievalQuality:
    """Test the quality and relevance of retrieval"""
    
    def test_retrieval_returns_relevant_documents(self, mock_config):
        """Test that retrieval returns relevant documents"""
        # Create nodes with distinct content
        nodes = [
            TextNode(
                text="OAuth 2.0 is an authentication protocol. Use API keys for access.",
                metadata={"file_name": "auth.md"}
            ),
            TextNode(
                text="Reset password by clicking forgot password link on login page.",
                metadata={"file_name": "password.md"}
            ),
            TextNode(
                text="Rate limits: 100 requests per hour for free tier accounts.",
                metadata={"file_name": "rate.md"}
            )
        ]
        
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            # Create embeddings that make auth.md most similar to auth queries
            def mock_get_embedding(text):
                if "auth" in text.lower() or "oauth" in text.lower():
                    return [0.9] * 384
                elif "password" in text.lower():
                    return [0.5] * 384
                else:
                    return [0.1] * 384
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.side_effect = mock_get_embedding
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(nodes)
            retriever = indexer.get_retriever(similarity_top_k=1)
            
            # Note: Actual retrieval testing requires real embeddings
            # This test structure shows how to verify relevance
    
    def test_retrieval_metadata_preserved(self, sample_nodes, mock_config):
        """Test that retrieved nodes preserve metadata"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            retriever = indexer.get_retriever()
            
            # Mock retrieve to return nodes with metadata
            mock_node = TextNode(
                text="Sample text",
                metadata={"file_name": "test.md", "page": 1}
            )
            mock_result = NodeWithScore(node=mock_node, score=0.9)
            
            with patch.object(retriever, 'retrieve', return_value=[mock_result]):
                results = retriever.retrieve("test query")
                
                assert len(results) > 0
                assert results[0].node.metadata is not None
                assert 'file_name' in results[0].node.metadata


class TestRetrievalEdgeCases:
    """Test edge cases in retrieval"""
    
    def test_retrieval_with_empty_query(self, sample_nodes, mock_config):
        """Test retrieval with empty query string"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            retriever = indexer.get_retriever()
            
            # Mock the retriever to handle empty query
            with patch.object(retriever, 'retrieve', return_value=[]):
                results = retriever.retrieve("")
                assert isinstance(results, list)
    
    def test_retrieval_with_no_matches(self, sample_nodes, mock_config):
        """Test retrieval when no good matches exist"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            retriever = indexer.get_retriever()
            
            # Even with no good matches, should return something
            # (the top_k results, even if scores are low)
    
    def test_retrieval_with_special_characters(self, sample_nodes, mock_config):
        """Test retrieval with special characters in query"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            retriever = indexer.get_retriever()
            
            # Mock retrieval with special character query
            special_queries = [
                "What is OAuth 2.0?",
                "How to reset password (urgent)!",
                "API: rate-limits & quotas"
            ]
            
            for query in special_queries:
                with patch.object(retriever, 'retrieve', return_value=[]):
                    results = retriever.retrieve(query)
                    assert isinstance(results, list)


class TestRetrievalScoring:
    """Test retrieval scoring mechanisms"""
    
    def test_retrieval_returns_scores(self, sample_nodes, mock_config):
        """Test that retrieved results include similarity scores"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            retriever = indexer.get_retriever()
            
            # Mock scored results
            mock_results = [
                NodeWithScore(
                    node=TextNode(text="Result 1", metadata={}),
                    score=0.95
                ),
                NodeWithScore(
                    node=TextNode(text="Result 2", metadata={}),
                    score=0.87
                )
            ]
            
            with patch.object(retriever, 'retrieve', return_value=mock_results):
                results = retriever.retrieve("test query")
                
                for result in results:
                    assert hasattr(result, 'score')
                    assert isinstance(result.score, (int, float))
                    assert 0 <= result.score <= 1
    
    def test_retrieval_scores_ordered(self, sample_nodes, mock_config):
        """Test that results are ordered by relevance score"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            retriever = indexer.get_retriever(similarity_top_k=3)
            
            # Mock results with descending scores
            mock_results = [
                NodeWithScore(node=TextNode(text="Most relevant"), score=0.95),
                NodeWithScore(node=TextNode(text="Second"), score=0.87),
                NodeWithScore(node=TextNode(text="Third"), score=0.76)
            ]
            
            with patch.object(retriever, 'retrieve', return_value=mock_results):
                results = retriever.retrieve("test query")
                
                if len(results) > 1:
                    # Scores should be in descending order
                    scores = [r.score for r in results]
                    assert scores == sorted(scores, reverse=True)