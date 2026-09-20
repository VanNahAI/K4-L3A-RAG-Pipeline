"""Task 4 - Chunk, embed, and index the standardized corpus in ChromaDB.

Task 5 imports :func:`embed_texts` and :func:`get_collection` from this module,
which guarantees that indexing and semantic search use the same model and
vector store configuration.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .contracts import validate_document
from .task3_convert_markdown import METADATA_END, METADATA_START


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 32

COLLECTION_NAME = "rag_documents"
INDEX_BATCH_SIZE = 128


@lru_cache(maxsize=1)
def _get_embedding_model():
    if EMBEDDING_PROVIDER != "sentence_transformers":
        raise ValueError(
            "Task 4 supports EMBEDDING_PROVIDER=sentence_transformers only; "
            f"received {EMBEDDING_PROVIDER!r}"
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "sentence-transformers is not installed. Run: "
            "python -m pip install -e \".[dev]\""
        ) from exc

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed non-empty texts with the single configured local provider."""
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every text passed to embed_texts must be non-empty")

    vectors = _get_embedding_model().encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    result = vectors.tolist()
    if any(len(vector) != EMBEDDING_DIM for vector in result):
        actual = len(result[0]) if result else 0
        raise ValueError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {actual}"
        )
    return result


def get_collection():
    """Open the persistent Chroma collection configured for cosine distance."""
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "chromadb is not installed. Run: python -m pip install -e \".[dev]\""
        ) from exc

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _parse_markdown(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Standardized Markdown is empty: {path}")

    if raw.startswith(METADATA_START):
        header_end = raw.find(METADATA_END, len(METADATA_START))
        if header_end < 0:
            raise ValueError(f"Unclosed metadata header: {path}")
        header = raw[len(METADATA_START) : header_end].strip()
        metadata = json.loads(header)
        content = raw[header_end + len(METADATA_END) :].strip()
    else:
        metadata = {}
        content = raw

    if not content:
        raise ValueError(f"Standardized Markdown has no content: {path}")
    return metadata, content


def load_documents() -> list[dict]:
    """Load normalized Markdown as contract-compliant documents."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        metadata, content = _parse_markdown(path)
        relative_path = path.relative_to(STANDARDIZED_DIR)
        fallback_type = (
            relative_path.parts[0]
            if len(relative_path.parts) > 1 and relative_path.parts[0] in {"legal", "news"}
            else "legal"
        )
        heading = next(
            (
                line.removeprefix("# ").strip()
                for line in content.splitlines()
                if line.startswith("# ")
            ),
            path.stem.replace("_", " "),
        )
        doc_type = str(metadata.get("doc_type") or fallback_type)
        if doc_type not in {"legal", "news"}:
            doc_type = fallback_type
        document = {
            "id": relative_path.as_posix(),
            "content": content,
            "metadata": {
                "source": str(metadata.get("source") or path.name),
                "title": str(metadata.get("title") or heading),
                "doc_type": doc_type,
                "url": metadata.get("url") or None,
            },
        }
        validate_document(document)
        documents.append(document)

    ids = [document["id"] for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate standardized document IDs detected")
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents recursively while retaining identity and metadata."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "langchain-text-splitters is not installed. Run: "
            "python -m pip install -e \".[dev]\""
        ) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks: list[dict] = []

    for document in documents:
        validate_document(document)
        split_texts = splitter.split_text(document["content"])
        for index, text in enumerate(split_texts):
            cleaned = text.strip()
            if not cleaned:
                continue
            chunk = {
                "id": f"{document['id']}::chunk-{index:04d}",
                "content": cleaned,
                "metadata": {**document["metadata"], "chunk_index": index},
            }
            validate_document(chunk, require_chunk=True)
            chunks.append(chunk)

    ids = [chunk["id"] for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs detected")
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Return chunks with embeddings, preserving all original fields."""
    if not chunks:
        return []

    embedded: list[dict] = []
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
        for chunk in batch:
            validate_document(chunk, require_chunk=True)
        vectors = embed_texts([chunk["content"] for chunk in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding provider returned an unexpected vector count")
        embedded.extend(
            {**chunk, "metadata": dict(chunk["metadata"]), "embedding": vector}
            for chunk, vector in zip(batch, vectors)
        )
    return embedded


def _chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Convert metadata values to scalar types accepted by Chroma."""
    return {
        "source": str(metadata["source"]),
        "title": str(metadata["title"]),
        "doc_type": str(metadata["doc_type"]),
        "url": str(metadata.get("url") or ""),
        "chunk_index": int(metadata["chunk_index"]),
    }


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Synchronize embedded chunks into Chroma without duplicate records."""
    collection = get_collection()
    current_ids = {chunk["id"] for chunk in chunks}
    if len(current_ids) != len(chunks):
        raise ValueError("Duplicate chunk IDs cannot be indexed")

    existing_ids = set(collection.get(include=[]).get("ids", []))
    stale_ids = sorted(existing_ids - current_ids)
    for start in range(0, len(stale_ids), INDEX_BATCH_SIZE):
        collection.delete(ids=stale_ids[start : start + INDEX_BATCH_SIZE])

    for start in range(0, len(chunks), INDEX_BATCH_SIZE):
        batch = chunks[start : start + INDEX_BATCH_SIZE]
        for chunk in batch:
            validate_document(chunk, require_chunk=True)
            vector = chunk.get("embedding")
            if not isinstance(vector, list) or len(vector) != EMBEDDING_DIM:
                raise ValueError(f"Invalid embedding for chunk {chunk['id']}")
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=[_chroma_metadata(chunk["metadata"]) for chunk in batch],
        )


def run_pipeline() -> None:
    """Load, chunk, embed, and index the complete standardized corpus."""
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No Markdown documents found in {STANDARDIZED_DIR}")
    chunks = chunk_documents(documents)
    if not chunks:
        raise RuntimeError("No non-empty chunks were produced")
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(
        f"Indexed {len(embedded_chunks)} chunks from {len(documents)} documents "
        f"with {EMBEDDING_MODEL}"
    )


if __name__ == "__main__":
    run_pipeline()
