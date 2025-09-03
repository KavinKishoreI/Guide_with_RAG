# 💻 Laptop Buying Guide — Retrieval-Augmented Generation (RAG)

This project is a simple **RAG-based assistant** that answers your laptop purchasing questions using a custom **PDF guide**.  
It uses **LangChain**, **Hugging Face embeddings**, and **ChromaDB** as the vector store.  
Optionally, you can use **Ollama** (local LLM) or Hugging Face models for answering queries.

---

## 🚀 Features
- Ingest your **Laptop Buying Guide PDF** into a vector database.
- Chunk the document for efficient retrieval.
- Ask natural language questions and get answers **grounded in your PDF**.
- Supports **GPU acceleration (CUDA)** if available.
- Works with:
  - Local **Ollama** server (Mistral, Llama, etc.)
  - Hugging Face `flan-t5-base` (default)

---

## 🛠️ Setup

### 1. Clone repo and install dependencies
```bash
git clone https://github.com/your-username/laptop-buying-guide-rag.git
cd laptop-buying-guide-rag
python -m venv .venv
.venv\Scripts\activate   # (Windows)
pip install -r requirements.txt
python main.py
