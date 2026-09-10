"""
Centralized configuration for the Hiver SpotifyCares AI Support Agent.

All tuneable parameters, paths, and provider settings are here.
Override via environment variables for flexibility.
"""
import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
EVAL_DATA_DIR = DATA_DIR / "eval"
RESULTS_DIR = PROJECT_ROOT / "results"

# ─── Raw Dataset ─────────────────────────────────────────────────────────────
TWCS_CSV_PATH = os.getenv(
    "TWCS_CSV_PATH",
    str(PROJECT_ROOT / "Dataset_1" / "twcs" / "twcs.csv"),
)
TARGET_BRAND = os.getenv("TARGET_BRAND", "SpotifyCares")

# ─── Processed Data ──────────────────────────────────────────────────────────
SPOTIFY_RAW_JSONL = PROCESSED_DATA_DIR / "spotify_raw.jsonl"
SPOTIFY_THREADS_JSONL = PROCESSED_DATA_DIR / "spotify_threads.jsonl"
SPLIT_MANIFEST_JSON = SPLITS_DIR / "manifest.json"
CHROMA_DB_DIR = PROCESSED_DATA_DIR / "spotify_embeddings"

# ─── Eval Data ───────────────────────────────────────────────────────────────
TRAINING_LABELS_JSONL = EVAL_DATA_DIR / "training_labels.jsonl"
DEV_SET_JSONL = EVAL_DATA_DIR / "dev_set.jsonl"
GOLDEN_TEST_SET_JSONL = EVAL_DATA_DIR / "golden_test_set.jsonl"

# ─── LLM Provider (configurable – never hard-code a specific model) ────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "gpt-4o")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # set via env
LLM_TEMPERATURE_CLASSIFY = float(os.getenv("LLM_TEMPERATURE_CLASSIFY", "0.1"))
LLM_TEMPERATURE_GENERATE = float(os.getenv("LLM_TEMPERATURE_GENERATE", "0.7"))

# ─── Retrieval ───────────────────────────────────────────────────────────────
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─── Twitter / Platform Constraints ─────────────────────────────────────────
TWITTER_CHAR_LIMIT = 280

# ─── Escalation ──────────────────────────────────────────────────────────────
ESCALATION_CONFIDENCE_THRESHOLD = float(
    os.getenv("ESCALATION_THRESHOLD", "0.6")
)

# ─── Chronological Split Boundaries ─────────────────────────────────────────
# Threads whose earliest message is before this date → training split.
# Between TRAIN_CUTOFF and DEV_CUTOFF → dev split.
# After DEV_CUTOFF → test split.
TRAIN_CUTOFF = "2017-11-10"   # exclusive upper bound for training (~60%)
DEV_CUTOFF = "2017-11-21"     # exclusive upper bound for dev (~15%)

# ─── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42
BOOTSTRAP_N = 1000
