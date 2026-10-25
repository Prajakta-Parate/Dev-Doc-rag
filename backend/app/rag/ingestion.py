import io
import zipfile
from pathlib import PurePosixPath
import re

from app.config import (
    CHUNK_LINES,
    CHUNK_OVERLAP,
    MAX_EXTRACTED_BYTES,
    MAX_FILE_BYTES,
)

SUPPORTED_EXTENSIONS = {
    ".md", ".txt", ".rst",
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".json", ".yaml", ".yml",
    ".html", ".css", ".sql",
    ".java", ".go", ".rs",
    ".toml", ".ini", ".sh",
}

IGNORED_DIRS = {
    ".git", ".venv", "venv", "env",
    "node_modules", "__pycache__", "dist",
    "build", ".next", ".idea", ".vscode",
    "site-packages",
}


def split_into_chunks(text: str, file_path: str) -> list[dict]:
    lines = text.splitlines()
    chunks = []
    start = 0

    while start < len(lines):
        end = min(start + CHUNK_LINES, len(lines))
        chunk_text = "\n".join(lines[start:end]).strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "file_path": file_path,
                "start_line": start + 1,
                "end_line": end,
            })

        if end == len(lines):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def read_zip_documents(zip_bytes: bytes) -> list[dict]:
    documents = []
    total_bytes = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue

            # Normalize Windows separators before checking the path.
            name = item.filename.replace("\\", "/")
            path = PurePosixPath(name)

            if path.is_absolute() or ".." in path.parts:
                continue

            if any(part in IGNORED_DIRS for part in path.parts):
                continue

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            if item.file_size > MAX_FILE_BYTES:
                continue

            total_bytes += item.file_size
            if total_bytes > MAX_EXTRACTED_BYTES:
                raise ValueError(
                    "Repository exceeds the extracted-size limit."
                )

            try:
                raw = archive.read(item)
                text = raw.decode("utf-8")
            except (UnicodeDecodeError, OSError, RuntimeError):
                continue

            # Skip empty files.
            if not text.strip():
                continue

            # documents.extend(
            #     split_into_chunks(text, str(path))   
            # )
            documents.extend(chunk_document(str(path), text))

    return documents



def chunk_document(
    path: str,
    text: str,
    max_lines: int = 60,
    overlap_lines: int = 15,
) -> list[dict]:
    """
    Split a document into chunks while preferring natural boundaries:
    Markdown headings, blank lines, and code declarations.

    Each chunk includes its source path and original line numbers.
    """
    lines = text.splitlines()

    if not lines:
        return []

    if max_lines < 1:
        raise ValueError("max_lines must be at least 1")

    if not 0 <= overlap_lines < max_lines:
        raise ValueError(
            "overlap_lines must be >= 0 and smaller than max_lines"
        )

    chunks = []
    start = 0
    total_lines = len(lines)

    heading_pattern = re.compile(r"^\s{0,3}#{1,6}\s+")
    declaration_pattern = re.compile(
        r"^\s*(?:"
        r"(?:async\s+)?def\s+|"
        r"class\s+|"
        r"function\s+|"
        r"export\s+(?:default\s+)?(?:async\s+)?"
        r"(?:function|class|const|let|var)\s+|"
        r"public\s+(?:static\s+)?(?:class|interface|void)\s+"
        r")"
    )

    

    while start < total_lines:
        hard_end = min(start + max_lines, total_lines)
        end = hard_end

        # If the boundary falls inside a fenced code block,
        # extend the chunk to include its closing fence.
        in_fence = False
        fence_char = None
        fence_length = 0

        for i in range(start, hard_end):
            stripped = lines[i].lstrip()
            match = re.match(r"(`{3,}|~{3,})", stripped)

            if not match:
                continue

            marker = match.group(1)

            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                in_fence = False
                fence_char = None
                fence_length = 0

        if in_fence:
            # Find the closing fence after the current chunk boundary.
            for i in range(hard_end, total_lines):
                stripped = lines[i].lstrip()
                match = re.match(r"(`{3,}|~{3,})", stripped)

                if (
                    match
                    and match.group(1)[0] == fence_char
                    and len(match.group(1)) >= fence_length
                ):
                    end = i + 1
                    break
            else:
                # Unclosed fence: include the remaining lines.
                end = total_lines

        if hard_end < total_lines:
            # Track fenced Markdown code blocks so we don't choose
            # a boundary inside a fenced example.
            in_fence = False
            fence_char = None
            fence_length = 0
            candidates = []

            for i in range(start, hard_end):
                stripped = lines[i].lstrip()

                fence_match = re.match(r"(`{3,}|~{3,})", stripped)

                if fence_match:
                    marker = fence_match.group(1)
                    char = marker[0]
                    length = len(marker)

                    if not in_fence:
                        in_fence = True
                        fence_char = char
                        fence_length = length
                    elif (
                        char == fence_char
                        and length >= fence_length
                    ):
                        in_fence = False
                        fence_char = None
                        fence_length = 0

                    continue

                if in_fence:
                    continue

                # Prefer boundaries closer to the end of the chunk.
                if heading_pattern.match(lines[i]):
                    candidates.append((i, 3))
                elif not lines[i].strip():
                    candidates.append((i, 2))
                elif declaration_pattern.match(lines[i]):
                    candidates.append((i, 1))

            # Avoid producing tiny chunks by considering boundaries
            # only in the latter half of the target window.
            lower_bound = start + max(1, max_lines // 2)
            candidates = [
                item for item in candidates
                if item[0] >= lower_bound
            ]

            if candidates:
                # Prefer headings, then blank lines, then declarations.
                # Within a priority, choose the latest boundary.
                boundary, _ = max(
                    candidates,
                    key=lambda item: (item[1], item[0]),
                )

                # For headings/declarations, leave that line for
                # the next chunk instead of separating it from its body.
                if (
                    heading_pattern.match(lines[boundary])
                    or declaration_pattern.match(lines[boundary])
                ):
                    end = boundary
                else:
                    end = boundary + 1

        # Guarantee progress even if a boundary is at the chunk start.
        end = max(start + 1, end)

        chunk_text = "\n".join(lines[start:end]).strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "file_path": str(path),
                "start_line": start + 1,
                "end_line": end,
            })

        if end >= total_lines:
            break

        
        # Overlap preserves context between adjacent chunks.
        next_start = max(start + 1, end - overlap_lines)

        # If the next chunk would start inside a fenced code block,
        # move its start to the line after the closing fence.
        in_fence = False
        fence_char = None
        fence_length = 0

        for i in range(next_start):
            stripped = lines[i].lstrip()
            match = re.match(r"(`{3,}|~{3,})", stripped)

            if not match:
                continue

            marker = match.group(1)

            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                in_fence = False
                fence_char = None
                fence_length = 0

        if in_fence:
            # Find the closing fence and start after it.
            for i in range(next_start, total_lines):
                stripped = lines[i].lstrip()
                match = re.match(r"(`{3,}|~{3,})", stripped)

                if (
                    match
                    and match.group(1)[0] == fence_char
                    and len(match.group(1)) >= fence_length
                ):
                    next_start = i + 1
                    break
            else:
                next_start = total_lines

        start = next_start


    return chunks
