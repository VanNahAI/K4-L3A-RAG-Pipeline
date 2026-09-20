"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau.
"""

from src.contracts import SearchResult
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[SearchResult]:
    """Trả về hybrid hoặc pageindex SearchResult.
    
    Luồng xử lý:
        1. Chạy semantic_search và lexical_search để lấy candidate chunks.
        2. Đánh giá độ tin cậy dựa trên cosine score gốc của dense retrieval:
           - Nếu best_dense_score < score_threshold: query có thể out-of-domain
             hoặc không khớp corpus -> thử PageIndex fallback.
           - Nếu PageIndex trả kết quả, dùng kết quả đó.
           - Nếu PageIndex gặp lỗi provider hoặc không có kết quả -> fallback về hybrid.
        3. Gộp bảng xếp hạng bằng RRF đúng một lần (nếu use_reranking=True).
    """
    candidate_k = max(top_k * 2, top_k)
    try:
        dense_results = semantic_search(query, top_k=candidate_k)
    except Exception:
        dense_results = []

    try:
        sparse_results = lexical_search(query, top_k=candidate_k)
    except Exception:
        sparse_results = []

    best_dense_score = dense_results[0]["score"] if dense_results else 0.0

    # Nếu score dưới threshold, thử PageIndex fallback trước
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback[:top_k]
        except Exception:
            # Nếu PageIndex provider lỗi, không crash mà tiếp tục dùng hybrid
            pass

    # Gộp thứ hạng bằng RRF đúng một lần
    if use_reranking:
        fused = rerank_rrf([dense_results, sparse_results], top_k=top_k)
        return fused[:top_k]

    return dense_results[:top_k]


if __name__ == "__main__":
    for result in retrieve("test query", top_k=3):
        print(result)
