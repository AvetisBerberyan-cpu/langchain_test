# RAG Q&A Assistant 🤖

An intelligent question-answering system for internal documentation using Retrieval-Augmented Generation (RAG) architecture. This CLI tool processes markdown/text documentation and provides accurate, context-aware answers to user questions.

## 🌟 Features

- **Semantic Search**: Find relevant information across multiple documents using advanced embeddings
- **Context-Aware Answers**: Generate accurate responses based on retrieved documentation
- **Persistent Index**: Build once, query many times - index is saved for fast subsequent queries
- **Source Attribution**: See which documents were used to generate each answer
- **Easy CLI Interface**: Simple command-line interface for quick queries
- **Automatic Document Processing**: Intelligently chunks and indexes all documentation

## 🏗️ Architecture

The system implements a complete RAG pipeline:

1. **Ingestion**: Loads and chunks markdown/text files from `./docs/`
2. **Embedding**: Converts text chunks into dense vector representations
3. **Indexing**: Stores embeddings in a searchable vector database
4. **Retrieval**: Finds the most relevant chunks for a given question
5. **Generation**: Uses an LLM to synthesize a natural language answer

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GPT models)
- 10-15 markdown or text documentation files

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd rag-qa-assistant
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# You can get one from: https://platform.openai.com/api-keys
```

Your `.env` file should look like:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 5. Add Your Documentation

Place your markdown or text files in the `docs/` directory:

```bash
# Create docs directory if it doesn't exist
mkdir -p docs

# Add your documentation files
# docs/
# ├── getting-started.md
# ├── authentication.md
# ├── api-reference.md
# └── ... (more files)
```

### 6. Run Your First Query

```bash
python qa.py --question "How do I reset my password?"
```

On first run, the system will:

- Load all documents from `./docs/`
- Chunk them into manageable pieces
- Generate embeddings
- Build and save the vector index

Subsequent queries will be much faster as they reuse the saved index.

## 💻 Usage

### Basic Query

```bash
python qa.py --question "What is the API rate limit?"
```

### Show Detailed Information

```bash
python qa.py --question "How to authenticate?" --verbose
```

This will show:

- Retrieved passage snippets
- Relevance scores
- Source file names

### Hide Source Attribution

```bash
python qa.py --question "What are the system requirements?" --no-sources
```

### Rebuild Index (After Adding New Documents)

```bash
python qa.py --rebuild
```

Then run your query:

```bash
python qa.py --question "Your question here"
```

### Short Form Options

```bash
# Use -q instead of --question
python qa.py -q "How do I get started?"

# Use -v instead of --verbose
python qa.py -q "What is OAuth?" -v
```

## 📁 Project Structure

```
rag-qa-assistant/
├── docs/                      # Your documentation files (markdown/text)
│   ├── file1.md
│   ├── file2.md
│   └── ...
│
├── src/                       # Source code
│   ├── __init__.py           # Package initialization
│   ├── ingestion.py          # Document loading and chunking
│   └── indexer.py            # Embedding and vector storage
│
├── storage/                   # Vector index (auto-generated)
│   ├── docstore.json         # Document store
│   ├── vector_store.json     # Vector embeddings
│   └── index_store.json      # Index metadata
│
├── qa.py                      # Main CLI interface
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (not in git)
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## ⚙️ Configuration

Edit `config.py` to customize behavior:

```python
# Document processing
CHUNK_SIZE = 512              # Size of text chunks (in tokens)
CHUNK_OVERLAP = 50            # Overlap between chunks

# Retrieval
TOP_K = 3                     # Number of chunks to retrieve

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"   # Can use: gpt-4, gpt-4-turbo

# Generation
LLM_TEMPERATURE = 0.1         # Lower = more focused answers
```

## 🔧 Advanced Usage

### Using Different LLM Models

In `config.py`, change:

```python
LLM_MODEL = "gpt-4"  # More capable but slower and more expensive
```

### Adjusting Retrieval

For more context (may be slower):

```python
TOP_K = 5  # Retrieve top 5 chunks instead of 3
```

For larger chunks:

```python
CHUNK_SIZE = 1024  # Larger chunks, fewer total chunks
```

### Debug Mode

Enable detailed logging:

```bash
# In config.py or environment
export LOG_LEVEL=DEBUG
python qa.py --question "Your question" --verbose
```

## 🧪 Testing

Test with sample questions:

```bash
# Test basic retrieval
python qa.py -q "What is this documentation about?"

# Test specific topic
python qa.py -q "How do I authenticate with the API?"

# Test with verbose output
python qa.py -q "What are the rate limits?" -v
```

## 🛠️ Troubleshooting

### "No module named 'llama_index'"

```bash
pip install -r requirements.txt
```

### "OPENAI_API_KEY not found"

Make sure your `.env` file exists and contains:

```
OPENAI_API_KEY=sk-your-key-here
```

### "Documents directory './docs' not found"

```bash
mkdir docs
# Add your markdown files to the docs/ directory
```

### Slow First Query

This is normal! The first query needs to:

- Load all documents
- Generate embeddings (2-5 minutes for 10-15 files)
- Build and save the index

Subsequent queries will be much faster (1-3 seconds).

### Index Not Updating After Adding New Documents

Rebuild the index:

```bash
python qa.py --rebuild
python qa.py -q "Your question"
```

## 📊 Performance

**First Run (Building Index):**

- 10-15 documents: ~3-5 minutes
- Generates embeddings for all chunks
- Saves index for future use

**Subsequent Queries:**

- Average: 1-3 seconds
- Loads existing index (fast)
- Only generates answer, no re-indexing

## 🔐 Security Notes

- Never commit your `.env` file (it's in `.gitignore`)
- Keep your OpenAI API key secure
- Monitor your API usage at https://platform.openai.com/usage

## 📝 How It Works

1. **Document Loading**: Reads all `.md` and `.txt` files from `docs/`
2. **Chunking**: Splits documents into 512-token chunks with 50-token overlap
3. **Embedding**: Converts chunks to 384-dimensional vectors using sentence-transformers
4. **Indexing**: Stores vectors in a searchable ChromaDB vector store
5. **Query Processing**:
   - Embeds user question
   - Finds top-3 most similar chunks (cosine similarity)
   - Constructs prompt with retrieved context
   - Sends to GPT-3.5 for answer generation
6. **Response**: Returns natural language answer with source attribution

## 🎯 Design Decisions

**Why LlamaIndex?**

- Purpose-built for RAG applications
- Simpler than LangChain for this use case
- Excellent documentation and abstractions
- Built-in query engine handles retrieval + generation

**Why sentence-transformers embeddings?**

- Fast inference
- Good semantic understanding
- No API costs (runs locally)
- Proven performance for retrieval tasks

**Why ChromaDB?**

- Lightweight (no external dependencies)
- Fast for small to medium datasets
- Easy to persist and reload
- Good Python integration

## 🚀 Future Enhancements

Potential improvements:

- [ ] Add support for PDF documents
- [ ] Implement hybrid search (semantic + keyword)
- [ ] Add conversation history
- [ ] Create web UI with Streamlit
- [ ] Add support for local LLMs (Ollama)
- [ ] Implement re-ranking for better accuracy
- [ ] Add metadata filtering
- [ ] Export conversation logs

## 📚 Resources

- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## 📄 License

This project is provided as-is for educational and internal use.

## 🤝 Contributing

Feel free to submit issues or pull requests for improvements.

## 📧 Contact

For questions or support, please contact [your-email@example.com]

---
