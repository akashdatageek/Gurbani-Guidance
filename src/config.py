import os
from dotenv import load_dotenv

load_dotenv()

# Model
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))

# Paths
DATA_DIR = os.getenv("DATA_DIR", "data")
RAW_ANGS_DIR = os.path.join(DATA_DIR, "raw_angs")
SHABADS_FILE = os.path.join(DATA_DIR, "shabads.jsonl")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")

# Retrieval
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "12"))
WINDOW_OVERLAP = int(os.getenv("WINDOW_OVERLAP", "2"))
TOP_K = int(os.getenv("TOP_K", "8"))
DENSE_K = int(os.getenv("DENSE_K", "20"))
SPARSE_K = int(os.getenv("SPARSE_K", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))

# BaniDB
BANIDB_BASE = "https://api.banidb.com/v2"
BANIDB_USER_AGENT = (
    "GurbaniRAG/1.0 (scripture-study-tool; github.com/akashdatageek/gurbani-guidance)"
)
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "0.5"))

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# ChromaDB
CHROMA_COLLECTION = "sggs"
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
