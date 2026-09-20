"""Task 6 — BM25 lexical search over the chunks produced by Task 4."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any

from .contracts import ChunkMetadata, SearchResult, validate_document


# Tests or callers may inject an already-built corpus here. When it is empty,
# lexical_search lazily loads and chunks the standardized documents via Task 4.
CORPUS: list[dict] = []
_INDEX_SIGNATURE: tuple[tuple[str, str], ...] | None = None
_INDEX: Any = None


def _tokenize(text: str) -> list[str]:
    """Tokenize case- and accent-insensitively for Vietnamese user queries."""
    decomposed = unicodedata.normalize("NFKD", text).casefold()
    normalized = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    ).replace("đ", "d")
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


@lru_cache(maxsize=1)
def _load_shared_corpus() -> list[dict]:
    """Build the exact same chunks as Task 4 without embedding them."""
    from .task4_chunking_indexing import chunk_documents, load_documents

    return chunk_documents(load_documents())


def _active_corpus() -> list[dict]:
    return CORPUS if CORPUS else _load_shared_corpus()


def build_bm25_index(corpus: list[dict]):
    """Create a BM25Okapi index from contract-compliant Task 4 chunks."""
    if not corpus:
        raise ValueError("Cannot build a BM25 index from an empty corpus")

    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:  # pragma: no cover - depends on runtime setup
        raise RuntimeError(
            'rank-bm25 is not installed. Run: python -m pip install -e ".[dev]"'
        ) from exc

    tokenized_corpus: list[list[str]] = []
    for item in corpus:
        validate_document(item, require_chunk=True)
        tokens = _tokenize(item["content"])
        if not tokens:
            # Keep positional alignment with the Task 4 corpus. The sentinel
            # cannot be produced by _tokenize, so this chunk will never match.
            tokens = ["\0"]
        tokenized_corpus.append(tokens)
    return BM25Okapi(tokenized_corpus)


def _get_cached_index(corpus: list[dict]):
    """Reuse the index until a chunk ID or its content changes."""
    global _INDEX, _INDEX_SIGNATURE

    signature = tuple(
        (str(item.get("id", "")), str(item.get("content", ""))) for item in corpus
    )
    if _INDEX is None or signature != _INDEX_SIGNATURE:
        _INDEX = build_bm25_index(corpus)
        _INDEX_SIGNATURE = signature
    return _INDEX


def _clean_metadata(raw_metadata: dict[str, Any]) -> ChunkMetadata:
    """Copy metadata into the strict public contract used by all retrievers."""
    doc_type = raw_metadata.get("doc_type")
    if doc_type not in {"legal", "news"}:
        raise ValueError(f"Invalid chunk doc_type: {doc_type!r}")
    return {
        "source": str(raw_metadata["source"]),
        "title": str(raw_metadata["title"]),
        "doc_type": doc_type,
        "url": str(raw_metadata["url"]) if raw_metadata.get("url") else None,
        "chunk_index": int(raw_metadata["chunk_index"]),
    }


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return unique BM25 SearchResults ordered by descending BM25 score."""
    if (
        not isinstance(query, str)
        or not query.strip()
        or isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k <= 0
    ):
        return []

    query_tokens = _tokenize(query)
    corpus = _active_corpus()
    if not query_tokens or not corpus:
        return []

    bm25 = _get_cached_index(corpus)
    scores = bm25.get_scores(query_tokens)
    query_terms = set(query_tokens)

    # Only return chunks with a real lexical match. BM25Okapi can assign zero
    # IDF to terms in tiny corpora, so filtering solely on score would wrongly
    # discard a valid match (for example, one matching document out of two).
    candidates = [
        index
        for index in range(len(corpus))
        if query_terms.intersection(bm25.doc_freqs[index])
    ]
    candidates.sort(key=lambda index: (-float(scores[index]), index))

    results: list[SearchResult] = []
    seen_ids: set[str] = set()
    for index in candidates:
        item = corpus[index]
        item_id = str(item["id"])
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        result: SearchResult = {
            "id": item_id,
            "content": str(item["content"]),
            "score": float(scores[index]),
            "metadata": _clean_metadata(item["metadata"]),
            "retrieval_method": "bm25",
        }
        validate_document(result, require_chunk=True)
        results.append(result)
        if len(results) == top_k:
            break

    return results


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    for result in lexical_search("phương thức xét tuyển đại học", top_k=3):
        print(f"{result['score']:.4f} | {result['metadata']['title']}")
