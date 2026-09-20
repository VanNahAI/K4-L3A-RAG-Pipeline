"""Task 7 — Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from typing import Any

from .contracts import SearchResult, validate_document, validate_search_results


def _validate_ranked_item(item: object) -> dict[str, Any]:
    """Validate the fields RRF must preserve without mutating the input."""
    validate_document(item, require_chunk=True)
    assert isinstance(item, dict)  # narrowed by validate_document

    score = item.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("ranked item score must be numeric")
    method = item.get("retrieval_method")
    if method not in {"dense", "bm25", "pageindex"}:
        if method == "hybrid":
            raise ValueError("RRF input is already fused; fuse ranked lists only once")
        raise ValueError(f"invalid retrieval_method: {method!r}")
    return item


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse rankings with ``sum(1 / (k + rank))``, where rank starts at 1."""
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        return []
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    if not isinstance(ranked_lists, list):
        raise ValueError("ranked_lists must be a list of ranked lists")

    scores: dict[str, float] = {}
    items: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0

    for ranked_list in ranked_lists:
        if not isinstance(ranked_list, list):
            raise ValueError("each ranking must be a list")
        seen_in_list: set[str] = set()
        for rank, raw_item in enumerate(ranked_list, start=1):
            item = _validate_ranked_item(raw_item)
            item_id = item["id"]

            # A malformed source ranking must not boost the same ID twice.
            if item_id in seen_in_list:
                continue
            seen_in_list.add(item_id)

            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            if item_id not in items:
                items[item_id] = item
                first_seen[item_id] = seen_counter
                seen_counter += 1

    ranked_ids = sorted(
        scores,
        key=lambda item_id: (-scores[item_id], first_seen[item_id], item_id),
    )

    results: list[SearchResult] = []
    for item_id in ranked_ids[:top_k]:
        source = items[item_id]
        result: SearchResult = {
            "id": item_id,
            "content": str(source["content"]),
            "score": float(scores[item_id]),
            "metadata": dict(source["metadata"]),
            "retrieval_method": "hybrid",
        }
        results.append(result)

    validate_search_results(results, top_k=top_k, expected_method="hybrid")
    return results


if __name__ == "__main__":
    print("Run pytest tests/test_contracts.py::test_rrf_uses_rank_deduplicates_and_marks_hybrid -q")
