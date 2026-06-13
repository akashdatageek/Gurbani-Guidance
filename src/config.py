import os
from dotenv import load_dotenv

load_dotenv()

# ── Models ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "claude-haiku-4-5-20251001")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2500"))

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", "data")
RAW_ANGS_DIR = os.path.join(DATA_DIR, "raw_angs")
SHABADS_FILE = os.path.join(DATA_DIR, "shabads.jsonl")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")

# ── Retrieval ────────────────────────────────────────────────────────────────
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "12"))
WINDOW_OVERLAP = int(os.getenv("WINDOW_OVERLAP", "2"))
TOP_K = int(os.getenv("TOP_K", "8"))
DENSE_K = int(os.getenv("DENSE_K", "20"))
SPARSE_K = int(os.getenv("SPARSE_K", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))
# Minimum cosine similarity for a result to be considered "relevant"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

# ── History limits ───────────────────────────────────────────────────────────
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "10"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "4000"))  # per message

# ── BaniDB ───────────────────────────────────────────────────────────────────
BANIDB_BASE = "https://api.banidb.com/v2"
BANIDB_USER_AGENT = (
    "GurbaniRAG/1.0 (scripture-study-tool; github.com/akashdatageek/gurbani-guidance)"
)
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "0.5"))

# ── CORS ─────────────────────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "sggs"
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# ── Rate limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_WINDOW = float(os.getenv("RATE_LIMIT_WINDOW", "60.0"))   # seconds
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "10"))              # req per window
