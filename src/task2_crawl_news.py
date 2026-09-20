"""Task 2 — Crawl tin tức về tuyển sinh đại học.

Mỗi bài viết được lưu thành một tệp JSON UTF-8 trong
``data/landing/news`` với đầy đủ URL, tiêu đề, thời điểm crawl và nội dung
Markdown. Chạy lại module sẽ cập nhật đúng năm tệp hiện có, không sinh bản
trùng lặp.
"""

import asyncio
import json
import re
import unicodedata
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Năm bài bao quát các khía cạnh: phương thức xét tuyển, chỉ tiêu, học phí và
# điểm chuẩn. Các URL đều là bài viết công khai, không yêu cầu đăng nhập.
ARTICLE_URLS = [
    "https://vnexpress.net/chot-quy-che-tuyen-sinh-dai-hoc-2025-4854295.html",
    "https://vnexpress.net/dai-hoc-kinh-te-quoc-dan-mo-hai-chuong-trinh-moi-tang-gan-600-chi-tieu-4867221.html",
    "https://vnexpress.net/chi-tieu-tuyen-sinh-nam-2025-cua-truong-si-quan-luc-quan-2-truong-si-quan-cong-binh-va-truong-si-quan-thong-tin-4856249.html",
    "https://vnexpress.net/hoc-phi-hon-100-dai-hoc-nam-hoc-2025-2026-chi-tiet-nhat-4914465.html",
    "https://vnexpress.net/diem-chuan-dai-hoc-bach-khoa-ha-noi-hust-2025-chinh-xac-nhat-4928933.html",
]


class _ArticleParser(HTMLParser):
    """Extract VnExpress's main article as compact, retrieval-friendly Markdown."""

    _VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.parts: list[str] = []
        self._article_depth = 0
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if not self._article_depth and tag == "article" and "fck_detail" in classes:
            self._article_depth = 1
            return
        if not self._article_depth:
            if tag == "meta" and attributes.get("property") == "og:title":
                self.title = (attributes.get("content") or "").strip()
            return

        if tag not in self._VOID_TAGS:
            self._article_depth += 1
        if tag in {"script", "style", "button", "svg", "noscript"}:
            self._skip_depth += 1
        if self._skip_depth:
            return
        if tag in {"p", "div", "section", "figure", "table", "tr"}:
            self.parts.append("\n\n")
        elif tag in {"h1", "h2", "h3", "h4"}:
            self.parts.append(f"\n\n{'#' * int(tag[1])} ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "br":
            self.parts.append("\n")
        elif tag in {"th", "td"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if not self._article_depth:
            return
        if tag in {"script", "style", "button", "svg", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if not self._skip_depth and tag in {"p", "h1", "h2", "h3", "h4", "li", "tr"}:
            self.parts.append("\n")
        self._article_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._article_depth and not self._skip_depth:
            text = re.sub(r"\s+", " ", data).strip()
            if text:
                if self.parts and not self.parts[-1].endswith((" ", "\n", "| ")):
                    self.parts.append(" ")
                self.parts.append(text)

    @property
    def markdown(self) -> str:
        content = "".join(self.parts)
        content = re.sub(r"[ \t]+\n", "\n", content)
        content = re.sub(r"\n{3,}", "\n\n", content)
        return content.strip()


async def crawl_article(url: str) -> dict[str, str]:
    """Crawl one public article and return the landing-data schema."""
    import requests

    def fetch() -> str:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; university-admissions-rag/1.0)"},
            timeout=30,
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        return response.text

    html = await asyncio.to_thread(fetch)
    parser = _ArticleParser()
    parser.feed(html)
    title = parser.title
    content = parser.markdown
    title = unicodedata.normalize("NFC", title)
    content = unicodedata.normalize("NFC", content)
    if not title:
        raise ValueError("article title is empty")
    if len(content) < 200:
        raise ValueError("article content is missing or unexpectedly short")

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content,
    }


async def crawl_all() -> None:
    """Crawl all configured articles and save each as one deterministic JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = await asyncio.gather(
        *(crawl_article(url) for url in ARTICLE_URLS),
        return_exceptions=True,
    )

    saved = 0
    for index, (url, result) in enumerate(zip(ARTICLE_URLS, results), 1):
        if isinstance(result, BaseException):
            print(f"Failed: {url} — {result}")
            continue

        output = DATA_DIR / f"article_{index:02d}.json"
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved += 1
        print(f"Saved: {output}")

    if saved != len(ARTICLE_URLS):
        raise RuntimeError(f"Only saved {saved}/{len(ARTICLE_URLS)} articles")


if __name__ == "__main__":
    asyncio.run(crawl_all())
