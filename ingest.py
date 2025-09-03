
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------ Config ------------------
DATA = {
    "ML": {
        "pdf_path": "Machine Learning Roadmap.pdf",
        "chroma_dir": ".ml/chroma_db"
    },
    "FSD": {
        "pdf_path": "web-roadmap.pdf",
        "chroma_dir": ".fsd/chroma_db"
    },
    "APP": {
        "pdf_path": "appdev.pdf",
        "chroma_dir": ".app/chroma_db"
    }
}
EMBED_MODEL = "BAAI/bge-small-en"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
# -------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

for key, cfg in DATA.items():
    print(f"\n--- Creating embeddings for {key} ---")
    loader = PyPDFLoader(cfg["pdf_path"])
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"Total chunks for {key}: {len(chunks)}")

    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=cfg["chroma_dir"])
    vectorstore.persist()
    print(f"✅ Embeddings stored in {cfg['chroma_dir']}")
