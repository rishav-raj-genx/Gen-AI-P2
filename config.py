"""GramHealth AI - Configuration module.
Rural Healthcare RAG Assistant.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
SOURCES_JSON = DATA_DIR / "sources.json"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure runtime directories exist
for p in [DOCUMENTS_DIR, VECTORSTORE_DIR, RESULTS_DIR, DOCS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Embedding Configuration
# Multilingual E5 small maps queries and passages to a 384-dimensional dense vector space.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
NORMALIZE_EMBEDDINGS = True

# Chunking Configurations
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Benchmark configurations for evaluation lab
CHUNKING_BENCHMARK_CONFIGS = [
    {"name": "Fixed-800-0", "method": "fixed", "chunk_size": 800, "chunk_overlap": 0},
    {"name": "Recursive-500-100", "method": "recursive", "chunk_size": 500, "chunk_overlap": 100},
    {"name": "Recursive-800-150", "method": "recursive", "chunk_size": 800, "chunk_overlap": 150},
    {"name": "Recursive-1200-200", "method": "recursive", "chunk_size": 1200, "chunk_overlap": 200},
]

# Retrieval Configuration
TOP_K = 4
SIMILARITY_THRESHOLD = 0.83  # Calibrated for multilingual-e5-small (in-domain > 0.84, out-of-domain < 0.825)

# LLM Generation Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_REQUEST_TIMEOUT = 45  # seconds

# Safety & Healthcare Guardrails
EMERGENCY_CONTACTS = {
    "National Emergency": "112",
    "Ambulance (Free)": "108",
    "Janani Shishu Suraksha Karyakram (Mother & Child)": "102",
    "Women Helpline": "1091",
    "National Tele-Mental Health (Tele-MANAS)": "14416",
    "Childline": "1098"
}

DISCLAIMER_TEXT = (
    "⚠️ **Educational Disclaimer:** GramHealth AI is an educational healthcare assistant "
    "developed for informational guidance only. It does NOT provide medical diagnosis, "
    "clinical prescriptions, or formal medical advice. In any acute or life-threatening emergency, "
    "please contact emergency services (108 / 112) or consult a qualified doctor at your nearest Primary Health Centre (PHC)."
)
