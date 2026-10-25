
import re
from functools import lru_cache
from uuid import uuid4

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import CHROMA_DIR, EMBEDDING_MODEL, TOP_K

COLLECTION_NAME = "developer_docs"
RRF_K = 60
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# Invalidated whenever the repository index changes.
_lexical_cache = None


def tokenize(text: str) -> list[str]:
    """Keep technical identifiers such as DATABASE_URL as single tokens."""
    return re.findall(
        r"[a-z_][a-z0-9_]*|[0-9]+",
        text.lower(),
    )


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(RERANK_MODEL)


def get_collection():
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def rebuild_index(chunks: list[dict]) -> int:
    global _lexical_cache

    collection = get_collection()

    existing_ids = collection.get(include=[])["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    # Invalidate the old keyword index.
    _lexical_cache = None

    model = get_embedding_model()
    batch_size = 64

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        embeddings = model.encode(
            [item["text"] for item in batch],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        collection.add(
            ids=[str(uuid4()) for _ in batch],
            documents=[item["text"] for item in batch],
            embeddings=embeddings,
            metadatas=[
                {
                    "file_path": item["file_path"],
                    "start_line": item["start_line"],
                    "end_line": item["end_line"],
                }
                for item in batch
            ],
        )

    # The next search rebuilds BM25 using the new documents.
    _lexical_cache = None

    return len(chunks)


def get_lexical_index():
    """Load all indexed chunks and build the BM25 index once."""
    global _lexical_cache

    if _lexical_cache is not None:
        return _lexical_cache

    collection = get_collection()

    stored = collection.get(
        include=["documents", "metadatas"],
    )

    ids = stored["ids"]
    documents = stored["documents"] or []
    metadatas = stored["metadatas"] or []

    if not documents:
        return None

    tokenized_documents = [tokenize(doc) for doc in documents]

    bm25 = BM25Okapi(tokenized_documents)

    _lexical_cache = {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "bm25": bm25,
    }

    return _lexical_cache


def retrieve_chunks(
    question: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Hybrid retrieval:
    1. Semantic search using ChromaDB.
    2. Keyword search using BM25.
    3. Merge rankings using Reciprocal Rank Fusion (RRF).
    """
    if top_k < 1:
        return []

    collection = get_collection()
    total_count = collection.count()

    if total_count == 0:
        return []

    candidate_count = min(top_k * 4, total_count)

    # -------------------------------------------------
    # 1. Semantic retrieval
    # -------------------------------------------------
    query_embedding = get_embedding_model().encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )

    semantic_ids = semantic_results["ids"][0]
    semantic_docs = semantic_results["documents"][0]
    semantic_metas = semantic_results["metadatas"][0]
    semantic_distances = semantic_results["distances"][0]

    # Save records by document ID so both retrieval methods
    # refer to the same underlying chunk.
    records = {}
    semantic_rankings = []

    for rank, (doc_id, document, metadata, distance) in enumerate(
        zip(
            semantic_ids,
            semantic_docs,
            semantic_metas,
            semantic_distances,
        ),
        start=1,
    ):
        records[doc_id] = {
            "text": document,
            "file_path": metadata["file_path"],
            "start_line": metadata["start_line"],
            "end_line": metadata["end_line"],
            "distance": distance,
        }
        semantic_rankings.append(doc_id)

    # -------------------------------------------------
    # 2. Keyword retrieval using BM25
    # -------------------------------------------------
    lexical = get_lexical_index()

    lexical_rankings = []

    if lexical is not None:
        query_tokens = tokenize(question)

        if query_tokens:
            scores = lexical["bm25"].get_scores(query_tokens)

            ranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )

            for index in ranked_indices[:candidate_count]:
                # Ignore documents with no positive BM25 match.
                if scores[index] <= 0:
                    continue

                doc_id = lexical["ids"][index]
                metadata = lexical["metadatas"][index]
                document = lexical["documents"][index]

                if doc_id not in records:
                    records[doc_id] = {
                        "text": document,
                        "file_path": metadata["file_path"],
                        "start_line": metadata["start_line"],
                        "end_line": metadata["end_line"],
                        # This chunk wasn't returned by semantic search.
                        "distance": None,
                    }

                lexical_rankings.append(doc_id)

    # -------------------------------------------------
    # 3. Reciprocal Rank Fusion
    # -------------------------------------------------
    fused_scores = {}

    for ranking in (semantic_rankings, lexical_rankings):
        for rank, doc_id in enumerate(ranking, start=1):
            fused_scores[doc_id] = (
                fused_scores.get(doc_id, 0.0)
                + 1.0 / (RRF_K + rank)
            )

    # Rank all hybrid candidates using Reciprocal Rank Fusion.
    ranked_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    if not ranked_ids:
        return []

    # Prepare question-document pairs for the cross-encoder.
    candidate_records = [
        records[doc_id]
        for doc_id in ranked_ids
    ]

    pairs = [
        (question, record["text"])
        for record in candidate_records
    ]

    # Score how relevant each chunk is to the question.
    reranker = get_reranker()
    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    # Sort candidates by cross-encoder relevance score.
    reranked = sorted(
        zip(ranked_ids, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    results = []

    for doc_id, score in reranked[:top_k]:
        record = records[doc_id].copy()
        record["retrieval_score"] = fused_scores[doc_id]
        record["rerank_score"] = float(score)
        results.append(record)

    return results
