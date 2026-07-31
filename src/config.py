import os
from dotenv import load_dotenv

load_dotenv()

# ── Provider ─────────────────────────────────────────────────────────────────
# Set PROVIDER=gemini to use Gemini instead of Claude
PROVIDER = os.getenv("PROVIDER", "anthropic").lower()  # "anthropic" | "gemini"

# ── Models ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "claude-haiku-4-5-20251001")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
GEMINI_CLASSIFIER_MODEL = os.getenv("GEMINI_CLASSIFIER_MODEL", "gemini-2.5-flash")

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = os.getenv("DATA_DIR", "data")
RAW_ANGS_DIR = os.path.join(DATA_DIR, "raw_angs")
SHABADS_FILE = os.path.join(DATA_DIR, "shabads.jsonl")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
PDF_PATH = os.getenv("PDF_PATH", os.path.join("src", "SriGuruGranthSahibJiDarpanEnglish.pdf"))

# ── Retrieval ────────────────────────────────────────────────────────────────
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "12"))
WINDOW_OVERLAP = int(os.getenv("WINDOW_OVERLAP", "2"))
TOP_K = int(os.getenv("TOP_K", "8"))
DENSE_K = int(os.getenv("DENSE_K", "20"))
SPARSE_K = int(os.getenv("SPARSE_K", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))
# Minimum cosine similarity for a result to be considered "relevant".
# Calibrated on the BaniDB corpus with bge-m3: in-scope questions (English,
# Gurmukhi, Hinglish, situational) score >= ~0.50; unrelated queries
# (tech/recipes/sports) score <= ~0.45. 0.47 splits with margin either side.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.47"))

# ── History limits ───────────────────────────────────────────────────────────
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "10"))
HISTORY_MAX_CHARS = int(os.getenv("HISTORY_MAX_CHARS", "4000"))  # per message

# ── BaniDB ───────────────────────────────────────────────────────────────────
BANIDB_BASE = "https://api.banidb.com/v2"
BANIDB_USER_AGENT = (
    "GurbaniRAG/1.0 (scripture-study-tool; github.com/akashdatageek/gurbani-guidance)"
)
CRAWL_DELAY = float(os.getenv("CRAWL_DELAY", "0.5"))

# ── Retrieval mode ───────────────────────────────────────────────────────────
# "local" (default) — hybrid dense+BM25 semantic retrieval over the corpus
#                     built ONCE from the BaniDB API (src.ingest → src.audit
#                     → src.embed). This is the full RAG logic: embeddings,
#                     similarity gating, RRF fusion, writer/raag filters.
# "banidb"          — live BaniDB search API per question (lexical full-word
#                     matching only — no semantic retrieval; degraded quality
#                     for situational/multilingual questions). Useful when a
#                     local index cannot be built.
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "local").lower()
# Results requested per BaniDB search call
BANIDB_SEARCH_RESULTS = int(os.getenv("BANIDB_SEARCH_RESULTS", "20"))

# ── CORS ─────────────────────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "sggs"
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "64"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")

# ── Rate limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_WINDOW = float(os.getenv("RATE_LIMIT_WINDOW", "60.0"))   # seconds
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "10"))              # req per window
