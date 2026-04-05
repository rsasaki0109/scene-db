"""Embedding-based semantic search for scene-db."""

import os
import sqlite3
import struct
from pathlib import Path

EMBEDDING_DIM = 384  # sentence-transformers default


def _get_embedding_model():
    """Load sentence-transformers model if available."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        return None


def _get_openai_embedder():
    """Get OpenAI embedding function if available."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        def embed(texts: list[str]) -> list[list[float]]:
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [d.embedding for d in resp.data]

        return embed
    except ImportError:
        return None


def _encode_embedding(vec: list[float]) -> bytes:
    """Encode float list to bytes for SQLite storage."""
    return struct.pack(f"{len(vec)}f", *vec)


def _decode_embedding(data: bytes) -> list[float]:
    """Decode bytes back to float list."""
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def ensure_embedding_table(conn: sqlite3.Connection) -> None:
    """Create the embeddings table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scene_embeddings (
            scene_id TEXT PRIMARY KEY REFERENCES scene_chunks(id),
            embedding BLOB NOT NULL
        )
    """)
    conn.commit()


def build_embeddings(conn: sqlite3.Connection) -> int:
    """Compute and store embeddings for all scenes. Returns count."""
    ensure_embedding_table(conn)

    # Try sentence-transformers first, then OpenAI
    model = _get_embedding_model()
    if model is not None:
        return _build_with_sentence_transformers(conn, model)

    openai_embed = _get_openai_embedder()
    if openai_embed is not None:
        return _build_with_openai(conn, openai_embed)

    raise RuntimeError(
        "No embedding backend available. Install sentence-transformers "
        "(pip install sentence-transformers) or set OPENAI_API_KEY."
    )


def _build_with_sentence_transformers(conn: sqlite3.Connection, model) -> int:
    """Build embeddings using sentence-transformers."""
    cursor = conn.execute("SELECT id, caption FROM scene_chunks")
    rows = cursor.fetchall()
    if not rows:
        return 0

    ids = [r[0] for r in rows]
    captions = [r[1] for r in rows]
    vectors = model.encode(captions)

    for scene_id, vec in zip(ids, vectors):
        conn.execute(
            "INSERT OR REPLACE INTO scene_embeddings (scene_id, embedding) VALUES (?, ?)",
            (scene_id, _encode_embedding(vec.tolist())),
        )
    conn.commit()
    return len(ids)


def _build_with_openai(conn: sqlite3.Connection, embed_fn) -> int:
    """Build embeddings using OpenAI API."""
    cursor = conn.execute("SELECT id, caption FROM scene_chunks")
    rows = cursor.fetchall()
    if not rows:
        return 0

    ids = [r[0] for r in rows]
    captions = [r[1] for r in rows]

    # Batch in groups of 100
    for i in range(0, len(captions), 100):
        batch_ids = ids[i : i + 100]
        batch_caps = captions[i : i + 100]
        vectors = embed_fn(batch_caps)
        for scene_id, vec in zip(batch_ids, vectors):
            conn.execute(
                "INSERT OR REPLACE INTO scene_embeddings (scene_id, embedding) VALUES (?, ?)",
                (scene_id, _encode_embedding(vec)),
            )
    conn.commit()
    return len(ids)


def semantic_search(
    conn: sqlite3.Connection, query: str, top_k: int = 10
) -> list[tuple[str, float]]:
    """Search scenes by semantic similarity. Returns list of (scene_id, score)."""
    # Encode query
    model = _get_embedding_model()
    if model is not None:
        query_vec = model.encode([query])[0].tolist()
    else:
        openai_embed = _get_openai_embedder()
        if openai_embed is not None:
            query_vec = openai_embed([query])[0]
        else:
            raise RuntimeError("No embedding backend available.")

    # Load all embeddings and compute similarity
    cursor = conn.execute("SELECT scene_id, embedding FROM scene_embeddings")
    results = []
    for scene_id, emb_blob in cursor.fetchall():
        vec = _decode_embedding(emb_blob)
        score = _cosine_similarity(query_vec, vec)
        results.append((scene_id, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
