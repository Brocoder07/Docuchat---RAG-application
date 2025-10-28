# 📚 DocuChat - RAG-based Document Q&A System

**A production-ready RAG (Retrieval-Augmented Generation) application that allows users to upload documents and query them conversationally using a local LLM.**

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 🎯 Features

- **📄 Multi-format Document Support**: PDF, TXT, DOCX, MD, XLSX
- **🤖 Local LLM Integration**: Powered by Ollama (llama3.2:1b-instruct-q4_1)
- **🔍 Intelligent Search**: ChromaDB vector store with semantic search
- **💬 Interactive Chat**: Streamlit-based UI with chat history
- **🔒 100% Private**: All processing happens locally
- **📊 Source Tracking**: Detailed provenance for all answers
- **⚡ Fast & Efficient**: Optimized chunking and retrieval

---

## 🏗️ Architecture

```
┌─────────────────┐
│   Streamlit UI  │
│   (Frontend)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI       │
│   (Backend API) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         RAG Pipeline                │
├─────────────────────────────────────┤
│ • Document Processor (Chunking)     │
│ • Embedding Manager (Sentence-BERT) │
│ • ChromaDB (Vector Store)           │
│ • LLM Integration (Ollama)          │
└─────────────────────────────────────┘
```

---

## 📋 Prerequisites

- **Python 3.10+**
- **Ollama** (for LLM inference)
- **Git**

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Brocoder07/Docuchat---RAG-application.git
cd docuchat
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

**Windows:**
```bash
# Download from https://ollama.ai/download
# Or use winget
winget install Ollama.Ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Mac:**
```bash
brew install ollama
```

### 5. Pull the LLM Model

```bash
ollama pull llama3.2:1b-instruct-q4_1
```

### 6. Start Ollama Service

```bash
# The service should auto-start
# To manually start:
ollama serve
```

### 7. Start the Backend API

```bash
python -m src.api
```

The API will be available at `http://localhost:8000`

### 8. Start the Frontend (New Terminal)

```bash
streamlit run src/frontend/app.py
```

The UI will open at `http://localhost:8501`

---

## 📖 Usage

### Upload Documents

1. Click on **"Upload Document"** in the sidebar
2. Select your file (PDF, TXT, DOCX, MD, or XLSX)
3. Wait for processing (automatic chunking and embedding)

### Ask Questions

1. Type your question in the chat input
2. Or use quick questions from the sidebar
3. Get AI-powered answers with source citations

### View Sources

- Expand **"Source Information"** to see:
  - Documents referenced
  - Confidence scores
  - Relevant text chunks

---

## 🔧 Configuration

### Modify Chunk Size

Edit `src/config.py`:

```python
@dataclass
class ChunkingConfig:
    chunk_size: int = 800  # Adjust this
    chunk_overlap: int = 100  # And this
```

### Change LLM Model

Edit `src/llm_integration.py`:

```python
self.model_name = "llama3.2:1b-instruct-q4_1"  # Change model
```

### Adjust API Settings

Edit `src/api/core/config.py`:

```python
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.md', '.xlsx']
```

---

## 🧪 Testing

### Test ChromaDB Migration

```bash
python scripts/test_chroma_migration.py
```

### Test Full RAG Pipeline

```bash
python scripts/test_full_migration.py
```

### Test Individual Components

```bash
# Test document processor
python -m src.document_processor

# Test LLM integration
python -m src.llm_integration

# Test RAG pipeline
python -m src.rag_pipeline
```

---

## 📁 Project Structure

```
docuchat/
├── src/
│   ├── api/                    # FastAPI backend
│   │   ├── routes/            # API endpoints
│   │   ├── services/          # Business logic
│   │   └── models/            # Pydantic schemas
│   ├── frontend/              # Streamlit UI
│   │   ├── components/        # UI components
│   │   ├── services/          # API client
│   │   └── utils/             # Session state
│   ├── config.py              # Global configuration
│   ├── document_processor.py  # Document parsing
│   ├── embedding_manager.py   # Embedding logic
│   ├── chroma_manager.py      # Vector store
│   ├── llm_integration.py     # LLM interface
│   └── rag_pipeline.py        # RAG orchestration
├── data/
│   ├── raw_documents/         # Uploaded files
│   └── vector_store/          # ChromaDB storage
├── scripts/                   # Utility scripts
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🐛 Troubleshooting

### Issue: "Ollama service not accessible"

**Solution:**
```bash
# Check if Ollama is running
ollama list

# If not, start it
ollama serve
```

### Issue: "Model not found"

**Solution:**
```bash
# Pull the required model
ollama pull llama3.2:1b-instruct-q4_1
```

### Issue: "ChromaDB initialization failed"

**Solution:**
```bash
# Delete the vector store and restart
rm -rf data/vector_store/
python -m src.api
```

### Issue: "Import errors"

**Solution:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

---

## 🔍 API Documentation

Once the backend is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

- `POST /documents/upload` - Upload document
- `POST /chat/query` - Ask question
- `GET /documents` - List documents
- `GET /health` - Health check

---

## 📊 Performance Metrics

- **Document Processing**: ~2-5 seconds per document
- **Query Response**: ~1-3 seconds (with Ollama)
- **Embedding Generation**: ~0.5 seconds per chunk
- **Vector Search**: <100ms

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Ollama** for local LLM inference
- **ChromaDB** for vector storage
- **Sentence Transformers** for embeddings
- **FastAPI** for the API framework
- **Streamlit** for the UI

---

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section

---

**Built with ❤️ for the AI/ML community**

