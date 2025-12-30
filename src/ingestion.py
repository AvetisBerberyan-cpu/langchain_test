from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from config import Config


class DocumentIngestion:
    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP

    def load_documents(self):
        """Load all markdown/text files from docs directory."""
        reader = SimpleDirectoryReader(
            input_dir=Config.DOCS_DIR,
            recursive=True,
            required_exts=[".md", ".txt"],
        )
        documents = reader.load_data()
        print(f"✓ Loaded {len(documents)} documents")
        return documents

    def chunk_documents(self, documents):
        """Split documents into chunks."""
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"✓ Created {len(nodes)} chunks")
        return nodes
