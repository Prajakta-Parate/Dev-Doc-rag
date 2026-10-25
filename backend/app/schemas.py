from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)


class Source(BaseModel):
    id: int
    file_path: str
    start_line: int
    end_line: int
    snippet: str
    distance: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int