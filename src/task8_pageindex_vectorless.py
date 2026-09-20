"""Task 8 — PageIndex vectorless fallback with persistent document-ID cache."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout, wait
from pathlib import Path
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv

from .contracts import ChunkMetadata, SearchResult, validate_search_results


load_dotenv()

ROOT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = ROOT_DIR / "data" / "standardized"
LANDING_LEGAL_DIR = ROOT_DIR / "data" / "landing" / "legal"
CACHE_PATH = ROOT_DIR / "pageindex_doc_ids.json"
PDF_CACHE_DIR = ROOT_DIR / "pageindex_pdfs"

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")


def _env_float(name: str, default: float, minimum: float) -> float:
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return default


PAGEINDEX_TIMEOUT = _env_float("PAGEINDEX_TIMEOUT", 30.0, 0.1)
PAGEINDEX_POLL_INTERVAL = _env_float("PAGEINDEX_POLL_INTERVAL", 1.0, 0.05)

T = TypeVar("T")


def _make_client():
    api_key = os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY).strip()
    if not api_key:
        raise RuntimeError("PAGEINDEX_API_KEY is not configured")
    try:
        from pageindex import PageIndexClient
    except ImportError as exc:  # pragma: no cover - depends on runtime setup
        raise RuntimeError(
            'pageindex is not installed. Run: python -m pip install -e ".[dev]"'
        ) from exc
    return PageIndexClient(api_key=api_key)


def _call_with_timeout(function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Bound an SDK call even though pageindex 0.2.8 exposes no timeout option."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(function, *args, **kwargs)
    try:
        return future.result(timeout=PAGEINDEX_TIMEOUT)
    except FutureTimeout as exc:
        future.cancel()
        raise TimeoutError(
            f"PageIndex call exceeded {PAGEINDEX_TIMEOUT:g} seconds"
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _load_cache() -> dict[str, dict[str, Any]]:
    if not CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid PageIndex cache: {CACHE_PATH}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("PageIndex cache must contain a JSON object")
    return {
        str(source): entry
        for source, entry in raw.items()
        if isinstance(entry, dict) and str(entry.get("doc_id", "")).strip()
    }


def _save_cache(cache: dict[str, dict[str, Any]]) -> None:
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unicode_font() -> Path:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError("No Unicode TrueType font found for Markdown-to-PDF conversion")


def _markdown_pdf(document: dict) -> Path:
    """Convert a standardized news document to the PDF format the API accepts."""
    try:
        from fpdf import FPDF
    except ImportError as exc:  # pragma: no cover - depends on runtime setup
        raise RuntimeError("fpdf2 is required to upload Markdown to PageIndex") from exc

    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output = PDF_CACHE_DIR / f"{Path(document['id']).stem}.pdf"
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("Unicode", fname=str(_unicode_font()))
    pdf.set_font("Unicode", size=9)
    for line in str(document["content"]).splitlines() or [""]:
        pdf.multi_cell(
            0,
            5,
            text=line or " ",
            new_x="LMARGIN",
            new_y="NEXT",
            wrapmode="CHAR",
        )
    pdf.output(str(output))
    return output


def _upload_path(document: dict) -> Path:
    """Reuse original legal PDFs; convert standardized news Markdown to PDF."""
    if document["metadata"]["doc_type"] == "legal":
        original = LANDING_LEGAL_DIR / f"{Path(document['id']).stem}.pdf"
        if original.is_file():
            return original
    return _markdown_pdf(document)


def upload_documents() -> None:
    """Upload new/changed corpus documents and persist source-to-ID mappings."""
    from .task4_chunking_indexing import load_documents

    client = _make_client()
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No standardized documents found in {STANDARDIZED_DIR}")

    cache = _load_cache()
    failures: list[str] = []
    for document in documents:
        source_key = str(document["id"])
        try:
            upload_path = _upload_path(document)
            fingerprint = _fingerprint(upload_path)
            cached = cache.get(source_key, {})
            if cached.get("fingerprint") == fingerprint and cached.get("doc_id"):
                continue

            response = _call_with_timeout(client.submit_document, str(upload_path))
            doc_id = str(response.get("doc_id", "")).strip()
            if not doc_id:
                raise RuntimeError("PageIndex upload response has no doc_id")

            cache[source_key] = {
                "doc_id": doc_id,
                "fingerprint": fingerprint,
                "source": document["metadata"]["source"],
                "title": document["metadata"]["title"],
                "doc_type": document["metadata"]["doc_type"],
                "url": document["metadata"]["url"],
            }
            _save_cache(cache)
            print(f"Uploaded: {source_key} -> {doc_id}")
        except Exception as exc:
            failures.append(f"{source_key}: {exc}")

    if failures:
        raise RuntimeError("Some PageIndex uploads failed:\n" + "\n".join(failures))


def _wait_for_retrieval(client: Any, doc_id: str, query: str) -> dict[str, Any]:
    submitted = _call_with_timeout(client.submit_query, doc_id, query)
    retrieval_id = str(submitted.get("retrieval_id", "")).strip()
    if not retrieval_id:
        raise RuntimeError("PageIndex query response has no retrieval_id")

    deadline = time.monotonic() + PAGEINDEX_TIMEOUT
    while time.monotonic() < deadline:
        response = _call_with_timeout(client.get_retrieval, retrieval_id)
        status = str(response.get("status", "")).lower()
        if status == "completed":
            return response
        if status in {"failed", "error", "cancelled"}:
            raise RuntimeError(f"PageIndex retrieval ended with status={status}")
        time.sleep(max(0.05, PAGEINDEX_POLL_INTERVAL))
    raise TimeoutError("Timed out waiting for PageIndex retrieval")


def _node_contents(node: dict[str, Any]) -> list[dict[str, Any]]:
    contents = node.get("relevant_contents")
    if isinstance(contents, list):
        return [item for item in contents if isinstance(item, dict)]
    return [node]


def _parse_response(
    response: dict[str, Any],
    cache_entry: dict[str, Any],
    document_order: int,
) -> list[tuple[float, int, SearchResult]]:
    """Parse the nested retrieved_nodes/relevant_contents API response."""
    nodes = response.get("retrieved_nodes", [])
    if not isinstance(nodes, list):
        return []

    parsed: list[tuple[float, int, SearchResult]] = []
    doc_id = str(cache_entry["doc_id"])
    local_rank = 0
    for node_index, raw_node in enumerate(nodes):
        if not isinstance(raw_node, dict):
            continue
        for content_index, content_item in enumerate(_node_contents(raw_node)):
            content = str(
                content_item.get("relevant_content")
                or content_item.get("content")
                or raw_node.get("text")
                or raw_node.get("content")
                or raw_node.get("summary")
                or ""
            ).strip()
            if not content:
                continue

            page = content_item.get("page_index", raw_node.get("page_index", 0))
            try:
                chunk_index = max(0, int(page))
            except (TypeError, ValueError):
                chunk_index = local_rank

            provider_score = content_item.get("score", raw_node.get("score"))
            score = (
                float(provider_score)
                if isinstance(provider_score, (int, float))
                and not isinstance(provider_score, bool)
                else 1.0 / (local_rank + 1)
            )
            metadata: ChunkMetadata = {
                "source": str(cache_entry.get("source") or "pageindex"),
                "title": str(cache_entry.get("title") or raw_node.get("title") or "PageIndex result"),
                "doc_type": cache_entry.get("doc_type")
                if cache_entry.get("doc_type") in {"legal", "news"}
                else "legal",
                "url": str(cache_entry["url"]) if cache_entry.get("url") else None,
                "chunk_index": chunk_index,
            }
            result: SearchResult = {
                "id": f"pageindex:{doc_id}:{node_index}:{content_index}",
                "content": content,
                "score": score,
                "metadata": metadata,
                "retrieval_method": "pageindex",
            }
            parsed.append((score, document_order, result))
            local_rank += 1
    return parsed


def _search_document(
    client: Any,
    entry: dict[str, Any],
    document_order: int,
    query: str,
) -> list[tuple[float, int, SearchResult]]:
    doc_id = str(entry["doc_id"])
    if not _call_with_timeout(client.is_retrieval_ready, doc_id):
        return []
    response = _wait_for_retrieval(client, doc_id, query)
    return _parse_response(response, entry, document_order)


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Return PageIndex results, or an empty list whenever the provider fails."""
    if (
        not isinstance(query, str)
        or not query.strip()
        or isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k <= 0
    ):
        return []

    try:
        client = _make_client()
        cache = _load_cache()
        if not cache:
            return []

        candidates: list[tuple[float, int, SearchResult]] = []
        ordered_entries = [cache[source_key] for source_key in sorted(cache)]
        executor = ThreadPoolExecutor(max_workers=min(4, len(ordered_entries)))
        futures = [
            executor.submit(_search_document, client, entry, order, query.strip())
            for order, entry in enumerate(ordered_entries)
        ]
        completed, pending = wait(futures, timeout=PAGEINDEX_TIMEOUT)
        for future in completed:
            try:
                candidates.extend(future.result())
            except Exception:
                continue
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]["id"]))
        results: list[SearchResult] = []
        seen_ids: set[str] = set()
        for _, _, result in candidates:
            if result["id"] in seen_ids:
                continue
            seen_ids.add(result["id"])
            results.append(result)
            if len(results) == top_k:
                break

        validate_search_results(
            results,
            top_k=top_k,
            expected_method="pageindex",
        )
        return results
    except Exception:
        return []


if __name__ == "__main__":
    upload_documents()
