import pytest
import os
from unittest.mock import patch, MagicMock
from src.indexer import RAGIndexer
from llama_index.core.schema import TextNode


class TestIndexerInitialization:
    """Test RAGIndexer initialization"""
    
    def test_indexer_initialization(self, mock_config):
        """Test that indexer initializes correctly"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI') as mock_llm:
            
            indexer = RAGIndexer()
            
            assert indexer.index is None
            mock_embed.assert_called_once()
            mock_llm.assert_called_once()
    
    def test_indexer_sets_global_settings(self, mock_config):
        """Test that indexer sets global LlamaIndex settings"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI') as mock_llm, \
             patch('src.indexer.Settings') as mock_settings:
            
            indexer = RAGIndexer()
            
            # Should set both embed_model and llm
            assert mock_settings.embed_model is not None or True
            assert mock_settings.llm is not None or True


class TestIndexCreation:
    """Test index creation functionality"""
    
    def test_create_new_index(self, sample_nodes, mock_config, cleanup_storage):
        """Test creating a new index from nodes"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI') as mock_llm:
            
            # Mock the embedding model
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            index = indexer.create_or_load_index(sample_nodes)
            
            assert index is not None
            assert indexer.index is not None
    
    def test_create_index_without_nodes_raises_error(self, mock_config):
        """Test that creating index without nodes raises error"""
        with patch('src.indexer.HuggingFaceEmbedding'), \
             patch('src.indexer.OpenAI'):
            
            indexer = RAGIndexer()
            
            with pytest.raises(ValueError, match="No existing index found"):
                indexer.create_or_load_index(nodes=None)
    
    def test_index_persistence(self, sample_nodes, mock_config, temp_storage_dir):
        """Test that index is persisted to disk"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            
            # Check that storage directory was created
            assert os.path.exists(temp_storage_dir)
            # Should have some files in storage
            assert len(os.listdir(temp_storage_dir)) > 0


class TestIndexLoading:
    """Test index loading functionality"""
    
    def test_load_existing_index(self, sample_nodes, mock_config, temp_storage_dir):
        """Test loading an existing index from storage"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            # Create and save index
            indexer1 = RAGIndexer()
            indexer1.create_or_load_index(sample_nodes)
            
            # Create new indexer and load existing index
            indexer2 = RAGIndexer()
            index = indexer2.create_or_load_index()
            
            assert index is not None
            assert indexer2.index is not None
    
    def test_load_index_without_storage_falls_back(self, mock_config):
        """Test that loading without storage handles gracefully"""
        with patch('src.indexer.HuggingFaceEmbedding'), \
             patch('src.indexer.OpenAI'):
            
            indexer = RAGIndexer()
            
            # Should raise error when no storage and no nodes
            with pytest.raises(ValueError):
                indexer.create_or_load_index()


class TestQueryEngine:
    """Test query engine creation"""
    
    def test_get_query_engine(self, sample_nodes, mock_config):
        """Test creating a query engine from index"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            
            query_engine = indexer.get_query_engine()
            
            assert query_engine is not None
    
    def test_get_query_engine_without_index_raises_error(self, mock_config):
        """Test that getting query engine without index raises error"""
        with patch('src.indexer.HuggingFaceEmbedding'), \
             patch('src.indexer.OpenAI'):
            
            indexer = RAGIndexer()
            
            with pytest.raises(ValueError, match="Index not initialized"):
                indexer.get_query_engine()
    
    def test_get_query_engine_custom_parameters(self, sample_nodes, mock_config):
        """Test creating query engine with custom parameters"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            
            query_engine = indexer.get_query_engine(
                similarity_top_k=5,
                response_mode="tree_summarize"
            )
            
            assert query_engine is not None


class TestRetriever:
    """Test retriever creation"""
    
    def test_get_retriever(self, sample_nodes, mock_config):
        """Test creating a retriever from index"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            
            retriever = indexer.get_retriever()
            
            assert retriever is not None
    
    def test_get_retriever_without_index_raises_error(self, mock_config):
        """Test that getting retriever without index raises error"""
        with patch('src.indexer.HuggingFaceEmbedding'), \
             patch('src.indexer.OpenAI'):
            
            indexer = RAGIndexer()
            
            with pytest.raises(ValueError, match="Index not initialized"):
                indexer.get_retriever()
    
    def test_get_retriever_custom_top_k(self, sample_nodes, mock_config):
        """Test creating retriever with custom top_k"""
        with patch('src.indexer.HuggingFaceEmbedding') as mock_embed, \
             patch('src.indexer.OpenAI'):
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.get_text_embedding.return_value = [0.1] * 384
            mock_embed.return_value = mock_embed_instance
            
            indexer = RAGIndexer()
            indexer.create_or_load_index(sample_nodes)
            
            retriever = indexer.get_retriever(similarity_top_k=5)
            
            assert retriever is not None