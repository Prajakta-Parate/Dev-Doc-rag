import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

CHROMA_DIR = Path(
    os.getenv(
        "CHROMA_DIR",
        str(PROJECT_ROOT / "backend" / "data" / "chroma"),
    )
)

CHROMA_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_EXTRACTED_BYTES = 100 * 1024 * 1024
MAX_FILE_BYTES = 1 * 1024 * 1024

CHUNK_LINES = 60
CHUNK_OVERLAP = 15
TOP_K = 5