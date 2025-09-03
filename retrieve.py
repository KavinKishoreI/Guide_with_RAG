print("Which knowledge base do you want to query?")
print("Machine Learning (ML) \nFull Stack Development (FSD)")
choice = input("Choose One (ML/FSD): ").strip().upper()


import os
import sys
import torch
import warnings
from typing import List, Tuple
import requests

warnings.filterwarnings("ignore")

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


# ------------------ Config ------------------
EMBED_MODEL = "BAAI/bge-small-en"
HF_LLM_MODEL = "google/flan-t5-base"
K = 5
USE_OLLAMA_BY_DEFAULT = True
# -------------------------------------------



if choice == "ML":
    CHROMA_DIR = ".ml/chroma_db"
elif choice == "FSD":
    CHROMA_DIR = ".fsd/chroma_db"

else:
    print("Invalid choice!")
    sys.exit()

# ------------------ Helper Functions ------------------
def has_ollama_server(host="http://localhost:11434"):
    try:
        r = requests.get(host + "/api/status", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def create_llm():
    if USE_OLLAMA_BY_DEFAULT and has_ollama_server():
        try:
            from langchain_community.llms import Ollama
            print("Using Ollama local server.")
            return Ollama(model="mistral")
        except Exception:
            print("Ollama client failed; falling back to HuggingFace.")

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
You are an assistant. Answer the question using ONLY the provided CONTEXT.
If unsure, provide your best answer based on the context.

Context:
{context}

Question: {question}

Answer (concise):
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

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cuda" }
    )
    if os.path.isdir(CHROMA_DIR) and Chroma is not None:
        print(f"Loading vectorstore from {CHROMA_DIR} ...")
        vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        return vs
    else:
        raise FileNotFoundError(f"No vectorstore found at {CHROMA_DIR}. Please run ingestion first.")

# ------------------ Main ------------------
def main(query: str):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": K})

    # Debug: show retrieved chunks
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

    print("\n=== Answer ===\n")
    print(answer.strip() if isinstance(answer, str) else str(answer))

    docs = result.get("source_documents", []) if isinstance(result, dict) else []
    sources = dedupe_sources(docs)
    print("\n=== Sources ===")
    for s, p in sources:
        if p is None:
            print(f"- {s}")
        else:
            print(f"- {s} (page {p})")


# ------------------ Run ------------------
if __name__ == "__main__":
    question = input("Enter your question: ").strip()
    main(question)
