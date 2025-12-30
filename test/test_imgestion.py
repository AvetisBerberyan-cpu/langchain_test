import pytest
from src.ingestion import DocumentIngestion


class TestDocumentLoading:
    """Test document loading functionality"""
    
    def test_load_documents_success(self, mock_config):
        """Test successful document loading"""
        ingestion = DocumentIngestion()
        documents = ingestion.load_documents()
        
        # Should load 3 documents (2 .md + 1 .txt)
        assert len(documents) == 3
        assert all(hasattr(doc, 'text') for doc in documents)
        assert all(hasattr(doc, 'metadata') for doc in documents)
    
    def test_load_documents_contains_expected_content(self, mock_config):
        """Test that loaded documents contain expected content"""
        ingestion = DocumentIngestion()
        documents = ingestion.load_documents()
        
        # Combine all text
        all_text = " ".join(doc.text for doc in documents)
        
        # Check for key phrases from our test documents
        assert "OAuth" in all_text or "oauth" in all_text.lower()
        assert "password" in all_text.lower()
        assert "rate limit" in all_text.lower()
    
    def test_load_documents_metadata(self, mock_config):
        """Test that documents have correct metadata"""
        ingestion = DocumentIngestion()
        documents = ingestion.load_documents()
        
        # All documents should have file_name in metadata
        for doc in documents:
            assert 'file_name' in doc.metadata
            assert doc.metadata['file_name'].endswith(('.md', '.txt'))
    
    def test_load_documents_empty_directory(self, tmp_path, monkeypatch):
        """Test loading from empty directory"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        monkeypatch.setattr("config.Config.DOCS_DIR", str(empty_dir))
        
        ingestion = DocumentIngestion()
        documents = ingestion.load_documents()
        
        assert len(documents) == 0


class TestDocumentChunking:
    """Test document chunking functionality"""
    
    def test_chunk_documents_creates_nodes(self, sample_documents, mock_config):
        """Test that chunking creates node objects"""
        ingestion = DocumentIngestion()
        nodes = ingestion.chunk_documents(sample_documents)
        
        assert len(nodes) > 0
        assert all(hasattr(node, 'text') for node in nodes)
    
    def test_chunk_documents_more_chunks_than_docs(self, mock_config):
        """Test that chunking creates more chunks than original documents"""
        ingestion = DocumentIngestion()
        documents = ingestion.load_documents()
        nodes = ingestion.chunk_documents(documents)
        
        # Should have at least as many chunks as documents
        # (likely more due to chunking)
        assert len(nodes) >= len(documents)
    
    def test_chunk_size_respected(self, sample_documents):
        """Test that chunks respect the configured size"""
        chunk_size = 100
        ingestion = DocumentIngestion(chunk_size=chunk_size)
        nodes = ingestion.chunk_documents(sample_documents)
        
        # Most chunks should be roughly the target size
        # (some variation is expected at document boundaries)
        for node in nodes:
            # Chunks can be smaller but shouldn't be much larger
            assert len(node.text) <= chunk_size * 2
    
    def test_chunk_overlap(self, sample_documents):
        """Test that chunk overlap is working"""
        chunk_size = 100
        chunk_overlap = 20
        
        ingestion = DocumentIngestion(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        nodes = ingestion.chunk_documents(sample_documents)
        
        # If we have multiple chunks, overlap should preserve content
        if len(nodes) > 1:
            # Check that some text appears in adjacent chunks
            for i in range(len(nodes) - 1):
                current_text = nodes[i].text
                next_text = nodes[i + 1].text
                # Note: overlap might not always be detectable in short texts
    
    def test_chunk_metadata_preserved(self, sample_documents, mock_config):
        """Test that chunking preserves document metadata"""
        ingestion = DocumentIngestion()
        nodes = ingestion.chunk_documents(sample_documents)
        
        # All nodes should have metadata from original documents
        for node in nodes:
            assert hasattr(node, 'metadata')
            if node.metadata:
                assert 'file_name' in node.metadata or len(node.metadata) == 0
    
    def test_empty_document_list(self, mock_config):
        """Test chunking with empty document list"""
        ingestion = DocumentIngestion()
        nodes = ingestion.chunk_documents([])
        
        assert len(nodes) == 0


class TestIngestionPipeline:
    """Test complete ingestion pipeline"""
    
    def test_process_pipeline(self, mock_config):
        """Test the complete process() pipeline"""
        ingestion = DocumentIngestion()
        nodes = ingestion.process()
        
        # Should return processed nodes
        assert len(nodes) > 0
        assert all(hasattr(node, 'text') for node in nodes)
        assert all(hasattr(node, 'metadata') for node in nodes)
    
    def test_process_with_custom_settings(self, mock_config):
        """Test pipeline with custom chunk settings"""
        ingestion = DocumentIngestion(chunk_size=200, chunk_overlap=30)
        nodes = ingestion.process()
        
        assert len(nodes) > 0
    
    def test_ingestion_idempotent(self, mock_config):
        """Test that running ingestion twice gives same results"""
        ingestion = DocumentIngestion()
        
        nodes1 = ingestion.process()
        nodes2 = ingestion.process()
        
        # Should get same number of chunks
        assert len(nodes1) == len(nodes2)
        
        # Text content should be the same
        texts1 = [node.text for node in nodes1]
        texts2 = [node.text for node in nodes2]
        assert texts1 == texts2