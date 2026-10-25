
import sys
from pathlib import Path

# Add the backend directory to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_DIR))

from app.rag.ingestion import chunk_document

sample = """# Authentication

The app uses JWT authentication.

## Token Expiration

Tokens expire after five minutes.

```python
def create_token(user_id):
    return encode(user_id)

def verify_token(token):
    return decode(token)
```

## Logout

The application clears the authentication cookie.
"""

chunks = chunk_document(
    "docs/auth.md",
    sample,
    max_lines=8,
    overlap_lines=2,
)

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ---")
    print({
    "file_path": chunk["file_path"],
    "start_line": chunk["start_line"],
    "end_line": chunk["end_line"],
    })
    print(chunk["text"])
