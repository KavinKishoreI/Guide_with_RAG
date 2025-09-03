import os
import sys
import torch
import warnings
from typing import List, Tuple
import requests

warnings.filterwarnings("ignore")

# ------------------ LangChain Imports ------------------
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline

# For ingestion
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ------------------ Config ------------------
PDF_FILE = "Laptop-buying-2025.pdf"      # your guide
CHROMA_DIR = ".laptop_guide/chroma_db"   # vectorstore folder

EMBED_MODEL = "BAAI/bge-small-en"
HF_LLM_MODEL = "google/flan-t5-base"
K = 5
USE_OLLAMA_BY_DEFAULT = True
# -------------------------------------------


# ------------------ Helper Functions ------------------
def has_ollama_server(host="http://localhost:11434"):
    try:
        r = requests.get(host + "/api/status", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def create_llm():
    """Create Ollama or HuggingFace pipeline as LLM"""
    if USE_OLLAMA_BY_DEFAULT and has_ollama_server():
        try:
            from langchain_community.llms import Ollama
            print("✅ Using Ollama local server.")
            return Ollama(model="mistral")
        except Exception:
            print("⚠️ Ollama client failed; falling back to HuggingFace.")

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text2text-generation",
                    model=HF_LLM_MODEL,
                    device=device,
                    max_new_tokens=512,
                    do_sample=False)
    try:
        from langchain_huggingface import HuggingFacePipeline
    except Exception:
        from langchain_community.llms import HuggingFacePipeline
    return HuggingFacePipeline(pipeline=pipe)


def build_prompt_template():
    template = """
You are a helpful assistant that gives clear, reliable laptop purchasing advice. 
Always use ONLY the provided CONTEXT. 
If the context does not fully answer the question, make the best possible suggestion based on the given data.

Context:
{context}

Question: {question}

Answer (concise, practical advice):
"""
    return PromptTemplate(template=template, input_variables=["context", "question"])


def dedupe_sources(docs) -> List[Tuple[str, str]]:
    seen = set()
    out = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        key = f"{src}::{page}"
        if key in seen:
            continue
        seen.add(key)
        out.append((src, page))
    return out

def create_vectorstore():
    """Load PDF, split into chunks, and persist embeddings in Chroma"""
    if not os.path.exists(PDF_FILE):
        print(f"❌ File not found: {PDF_FILE}")
        sys.exit(1)

    loader = PyPDFLoader(PDF_FILE)
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} pages from {PDF_FILE}")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"✂️ Created {len(chunks)} chunks")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    # Save to Chroma (auto-persist if directory is set)
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"✅ Vectorstore created at {CHROMA_DIR}")


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    if os.path.isdir(CHROMA_DIR) and Chroma is not None:
        print(f"📂 Loading vectorstore from {CHROMA_DIR} ...")
        vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        return vs
    else:
        raise FileNotFoundError(f"No vectorstore found at {CHROMA_DIR}. Run once to create it.")


def query_rag(query: str):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": K})

    docs = retriever.get_relevant_documents(query)
    if not docs:
        print("⚠️  No relevant chunks found!")
        return

    llm = create_llm()
    prompt = build_prompt_template()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    result = qa.invoke(query)
    answer = result.get("result") if isinstance(result, dict) else result

    print("\n=== Laptop Buying Guide Answer ===\n")
    print(answer.strip() if isinstance(answer, str) else str(answer))

    docs = result.get("source_documents", []) if isinstance(result, dict) else []
    sources = dedupe_sources(docs)
    print("\n=== Sources Used ===")
    for s, p in sources:
        if p is None:
            print(f"- {s}")
        else:
            print(f"- {s} (page {p})")


# ------------------ Run ------------------
if __name__ == "__main__":
    # Only create vectorstore if it doesn’t exist yet
    if not os.path.isdir(CHROMA_DIR):
        print("⚡ First run: Creating vectorstore from PDF...")
        create_vectorstore()
    else:
        print("📂 Using existing vectorstore...")

    while True:
        question = input("\n💻 Enter your laptop-related question (or 'exit' to quit): ").strip()
        if question.lower() == "exit":
            print("👋 Goodbye!")
            break
        query_rag(question)
