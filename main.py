from flask import Flask, render_template, request, jsonify
import os
import torch
import warnings
import requests

warnings.filterwarnings("ignore")

# LangChain imports
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
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from transformers import pipeline

# ---------------- CONFIG ----------------
EMBED_MODEL = "BAAI/bge-small-en"
HF_LLM_MODEL = "google/flan-t5-base"
K = 5
USE_OLLAMA_BY_DEFAULT = True

# Flask app
app = Flask(__name__)

# ---------------- GLOBAL CACHE ----------------
vectorstores = {}
llm = None

# ---------------- HELPER FUNCTIONS ----------------
def has_ollama_server(host="http://localhost:11434"):
    try:
        r = requests.get(host + "/api/status", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def create_llm():
    """Load the LLM once and cache"""
    global llm
    if llm is not None:
        return llm

    if USE_OLLAMA_BY_DEFAULT and has_ollama_server():
        try:
            from langchain_community.llms import Ollama
            print("Using Ollama local server.")
            llm = Ollama(model="mistral")
            return llm
        except Exception:
            print("Ollama client failed; falling back to HuggingFace.")

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text2text-generation",
        model=HF_LLM_MODEL,
        device=device,
        max_new_tokens=512,
        do_sample=False
    )
    try:
        from langchain_huggingface import HuggingFacePipeline
    except Exception:
        from langchain_community.llms import HuggingFacePipeline

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

def dedupe_sources(docs):
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

def load_vectorstore(choice):
    """Load vectorstore for a given KB"""
    global vectorstores
    if choice in vectorstores:
        return vectorstores[choice]

    # Chroma folder per knowledge base
    chroma_dirs = {
        "ML": ".ml/chroma_db",
        "FSD": ".fsd/chroma_db",
        "APP": ".app/chroma_db"
    }
    if choice not in chroma_dirs:
        raise ValueError("Invalid choice. Must be 'ML', 'FSD', or 'APP'")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    chroma_dir = chroma_dirs[choice]
    if os.path.isdir(chroma_dir) and Chroma is not None:
        print(f"Loading vectorstore from {chroma_dir} ...")
        vs = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
        vectorstores[choice] = vs
        return vs
    else:
        raise FileNotFoundError(f"No vectorstore found at {chroma_dir}. Run ingestion first.")

# ---------------- RAG QUERY ----------------
def query_rag(question: str, choice: str):
    try:
        vectorstore = load_vectorstore(choice)
        retriever = vectorstore.as_retriever(search_kwargs={"k": K})

        # Check if relevant docs exist
        docs = retriever.get_relevant_documents(question)
        if not docs:
            return {"answer": "No relevant information found.", "sources": [], "error": None}

        llm_instance = create_llm()

        # ---------------- PROMPT ----------------
        prompt = PromptTemplate(
            template="""
You are a helpful assistant. Use the context to answer the question, 
but feel free to rephrase or explain in a conversational tone.
Please keep the responses to not more than 100 words or so. 

Context:
{context}

Question: {question}

Answer (concise):
""",
            input_variables=["context", "question"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm_instance,          # use the correct LLM instance
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        # Invoke the QA chain
        result = qa.invoke(question)

        # Extract answer and docs
        answer = result.get("result") if isinstance(result, dict) else result
        result_docs = result.get("source_documents", docs)  # fallback to previously retrieved docs
        sources = dedupe_sources(result_docs)

        # Optional: log retrieved docs for debugging
        print(f"Retrieved {len(result_docs)} docs:")
        for d in result_docs:
            print(d.page_content[:200])

        return {
            "answer": answer.strip() if isinstance(answer, str) else str(answer),
            "sources": sources,
            "error": None
        }

    except Exception as e:
        return {"answer": None, "sources": [], "error": f"Error processing your question: {str(e)}"}

# ---------------- FLASK ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        choice = data.get('choice', '').strip().upper()

        if not question:
            return jsonify({"error": "Please enter a question"}), 400
        if choice not in ['ML', 'FSD', 'APP']:
            return jsonify({"error": "Please select ML, FSD, or APP"}), 400

        result = query_rag(question, choice)
        if result['error']:
            return jsonify({"error": result['error']}), 500

        return jsonify({
            "answer": result['answer'][:200],
            "sources": [{"source": s, "page": p} for s, p in result['sources']]
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "message": "RAG server is running"})

# ---------------- INITIALIZE ----------------
def initialize_app():
    print("Initializing RAG server...")
    try:
        create_llm()
        print("✓ LLM loaded")
        for choice in ['ML', 'FSD', 'APP']:
            try:
                load_vectorstore(choice)
                print(f"✓ {choice} vectorstore loaded")
            except FileNotFoundError:
                print(f"⚠️  {choice} vectorstore not found - will load on demand")
    except Exception as e:
        print(f"⚠️ Initialization warning: {e}")

if __name__ == '__main__':
    initialize_app()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
