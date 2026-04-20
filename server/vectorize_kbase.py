from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from knowledge_base_utils import build_documents_from_knowledge_base

# ---- Config ----
BASE_DIR = Path(__file__).resolve().parent
JSON_FILE = BASE_DIR / "kbase.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_STORE_PATH = BASE_DIR / "faiss_index"

# ---- Convert to LangChain Documents ----
documents = build_documents_from_knowledge_base(JSON_FILE)

# ---- Create Embeddings ----
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ---- Create FAISS Store ----
vector_store = FAISS.from_documents(documents, embeddings) 

# ---- Save FAISS Index ----
vector_store.save_local(str(FAISS_STORE_PATH))

print(f"✅ Embeddings created and saved to {FAISS_STORE_PATH}")
