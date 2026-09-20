"""Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from __future__ import annotations

from typing import Any

from .contracts import ChunkMetadata, SearchResult, validate_document
from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    collection = get_collection()

    n_results = top_k
    if hasattr(collection, "count"):
        try:
            total = collection.count()
            if total == 0:
                return []
            n_results = min(top_k, total)
        except Exception:
            pass

    query_vectors = embed_texts([query])
    if not query_vectors or not query_vectors[0]:
        return []
    query_vector = query_vectors[0]

    response = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]

    results: list[dict] = []
    seen_ids: set[str] = set()

    for item_id, content, raw_meta, distance in zip(ids, documents, metadatas, distances):
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        meta: dict[str, Any] = raw_meta or {}
        doc_type = meta.get("doc_type")
        if doc_type not in {"legal", "news"}:
            doc_type = "legal"

        clean_metadata: ChunkMetadata = {
            "source": str(meta.get("source") or "unknown"),
            "title": str(meta.get("title") or "unknown"),
            "doc_type": doc_type,
            "url": str(meta["url"]) if meta.get("url") else None,
            "chunk_index": int(meta.get("chunk_index", 0)),
        }

        # Convert cosine distance to cosine similarity
        score = float(max(0.0, 1.0 - float(distance)))

        search_result: SearchResult = {
            "id": str(item_id),
            "content": str(content),
            "score": score,
            "metadata": clean_metadata,
            "retrieval_method": "dense",
        }

        validate_document(search_result, require_chunk=True)
        results.append(search_result)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("test query", top_k=3):
        print(result)
