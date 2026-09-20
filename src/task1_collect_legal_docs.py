"""
Task 1 — Thu thập tài liệu tuyển sinh đại học.

Các tài liệu được lấy từ website chính thức của HUST, NEU và VinUni.
File gốc được lưu tại ``data/landing/legal/`` để Task 3 xử lý tiếp.

Chạy:
    python -m src.task1_collect_legal_docs
    python -m src.task1_collect_legal_docs --force
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TypedDict

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "K4-L3A-RAG-Pipeline/0.1 (educational data collection)"


class SourceDocument(TypedDict):
    filename: str
    title: str
    school: str
    url: str


SOURCES: tuple[SourceDocument, ...] = (
    {
        "filename": "hust_thong_tin_tuyen_sinh_dai_hoc_2026.pdf",
        "title": "Thông tin tuyển sinh đại học năm 2026",
        "school": "Đại học Bách khoa Hà Nội (HUST)",
        "url": "https://hust.edu.vn/uploads/sys/tuyen-sinh/2023_06/thong-tin-tuyen-sinh-dai-hoc-2026.pdf",
    },
    {
        "filename": "neu_tom_tat_thong_tin_tuyen_sinh_dai_hoc_chinh_quy_2026.pdf",
        "title": "Tóm tắt thông tin tuyển sinh đại học chính quy năm 2026",
        "school": "Đại học Kinh tế Quốc dân (NEU)",
        "url": "https://neu.edu.vn/wp-content/uploads/2025/12/Tom-tat-thong-tin-tuyen-sinh-nam-2026-DHCQ-1.pdf",
    },
    {
        "filename": "vinuni_quyet_dinh_ke_hoach_tuyen_sinh_dai_hoc_2026.pdf",
        "title": "Decision on the Issuance of the 2026 Undergraduate Admissions Plan",
        "school": "VinUniversity",
        "url": "https://admissions.vinuni.edu.vn/wp-content/uploads/sites/6/2025/04/Decision-on-the-Issuance-of-the-2026-Undergraduate-Admissions-Plan-of-VinUniversity-2-1.pdf",
    },
)


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def _validate_download(path: Path, source: SourceDocument) -> None:
    """Kiểm tra file tải về có đúng là PDF không bị rỗng."""
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Downloaded file is empty: {path.name}")

    with path.open("rb") as file:
        header = file.read(5)
    if path.suffix.lower() == ".pdf" and header != b"%PDF-":
        raise ValueError(
            f"Unexpected content for {source['school']}: {path.name} is not a PDF"
        )


def _download_one(source: SourceDocument, *, force: bool = False) -> Path:
    destination = DATA_DIR / source["filename"]
    if destination.exists() and not force:
        _validate_download(destination, source)
        print(f"Skip existing: {destination.name}")
        return destination

    temporary = destination.with_name(f".{destination.name}.part")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf"}

    try:
        with requests.get(
            source["url"],
            headers=headers,
            stream=True,
            timeout=(15, 120),
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        file.write(chunk)

        _validate_download(temporary, source)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(f"Downloaded: {destination.name} ({destination.stat().st_size:,} bytes)")
    return destination


def download_documents(*, force: bool = False) -> list[Path]:
    """Tải các tài liệu tuyển sinh từ nguồn chính thức."""
    downloaded: list[Path] = []
    for source in SOURCES:
        downloaded.append(_download_one(source, force=force))
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download official undergraduate admission documents."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again and overwrite existing files.",
    )
    args = parser.parse_args()

    setup_directory()
    download_documents(force=args.force)
    print(f"Collected {len(SOURCES)} admission documents in {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
