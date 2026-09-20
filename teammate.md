# 👥 Team Member & Task Assignment

## Thành viên nhóm

| Vai trò | Họ tên | GitHub |
|---------|--------|--------|
| 👑 Leader | Chu Văn Nhân | `NhanCV` |
| 🧑‍💻 Member | Nguyễn Khắc Quang | `Quang` |
| 🧑‍💻 Member | Dương Dương | `Duong` |

---

## Phân công nhiệm vụ

> **Nguyên tắc phân chia**: Mỗi thành viên phụ trách các file/module **riêng biệt** để tránh conflict khi push lên repository. Tuyệt đối không cùng chỉnh sửa một file trong cùng một thời điểm.

---

### 👑 Chu Văn Nhân — Leader

**Phạm vi**: Kiến trúc hệ thống, Retrieval Pipeline, Generation & UI tổng hợp, Evaluation & Báo cáo

#### Files phụ trách (KHÔNG ai được chỉnh sửa song song):
- `src/task9_retrieval_pipeline.py` — Kết hợp dense + BM25 + RRF + fallback logic
- `src/task10_generation.py` — Generation có citation, dispatch LLM provider
- `app.py` — Streamlit chatbot UI
- `src/contracts.py` — Schema & interface chung (chỉ Nhân được sửa)
- `pyproject.toml` — Dependency management
- `.env.example` — Cấu hình môi trường
- `tests/` — Contract & acceptance tests
- `group_project/evaluation/RESULT.md` — Báo cáo đánh giá

#### Nhiệm vụ cụ thể:
1. **Setup & kiến trúc**: Khởi tạo repo, review `src/contracts.py`, đảm bảo schema nhất quán giữa các task.
2. **Retrieval pipeline (Task 9)**:
   - Gọi `semantic_search()` + `lexical_search()` → `rerank_rrf()` → fallback dùng cosine score gốc
   - Calibrate `score_threshold` với query in-domain và out-of-domain
   - Đảm bảo RRF chỉ chạy **một lần**
3. **Generation có citation (Task 10)**:
   - Dispatch `LLM_PROVIDER` → OpenAI / Gemini / Anthropic Claude
   - Reorder chunks (không mất ID), format context với title/source
   - Safe refusal khi không đủ evidence
4. **Chatbot UI (`app.py`)**:
   - Hiển thị `answer`, `sources`, `retrieval_method`, `score`
   - Kết nối end-to-end từ input → retrieval → generation → display
5. **Evaluation & Tests**:
   - Tạo **golden dataset ≥ 15 Q&A** dựa trên corpus
   - Chạy 4 metric: faithfulness, answer relevance, context recall, context precision
   - So sánh A/B: dense-only vs hybrid + RRF
   - Điền đầy đủ `group_project/evaluation/RESULT.md`
   - Đảm bảo `pytest tests/test_contracts.py -q` và `pytest tests/test_acceptance.py -q` pass
   - Không gọi network hay API thật trong tests
6. **Review & merge PR** của các thành viên khác
7. **Hoàn thiện `group_project/individual/INDIVIDUAL_REPORT.md`** của bản thân

---

### 🧑‍💻 Nguyễn Khắc Quang — Data & Indexing

**Phạm vi**: Thu thập dữ liệu, Markdown conversion, Chunking & Embedding, Semantic Search

#### Files phụ trách (KHÔNG ai được chỉnh sửa song song):
- `src/task1_collect_legal_docs.py` — Thu thập tài liệu pháp lý (PDF/DOCX)
- `src/task3_convert_markdown.py` — Chuẩn hoá sang Markdown
- `src/task4_chunking_indexing.py` — Chunk, embed và index vào ChromaDB
- `src/task5_semantic_search.py` — Dense search từ ChromaDB
- `data/landing/legal/` — Thư mục chứa tài liệu pháp lý

#### Nhiệm vụ cụ thể:
1. **Thu thập tài liệu pháp lý (Task 1)**:
   - Tải tối thiểu **3 PDF/DOCX** vào `data/landing/legal/`
   - Mỗi file đặt tên rõ ràng, không trùng lặp
2. **Chuẩn hoá Markdown (Task 3)**:
   - Chuyển đổi PDF/DOCX → Markdown
   - Output chuẩn: mỗi doc có `id`, `content`, `metadata` theo schema trong `contracts.py`
3. **Chunking & Indexing (Task 4)**:
   - Implement `load_documents()`, `chunk_documents()`, `embed_texts()`, `embed_chunks()`, `index_to_vectorstore()`
   - Dùng **cùng embedding model và dimension** với Task 5
   - Đảm bảo chạy index lại không tạo dữ liệu trùng
4. **Semantic Search (Task 5)**:
   - Implement `semantic_search(query, top_k=10) -> list[SearchResult]`
   - Output đúng schema `SearchResult`, `retrieval_method="dense"`, sorted by score giảm dần
5. **Hoàn thiện `group_project/individual/INDIVIDUAL_REPORT.md`** của bản thân

> ⚠️ **Lưu ý tránh conflict**: Task 3 và Task 4 chia sẻ thư mục `data/processed/`. Quang tạo file mới, **KHÔNG xoá** file cũ của người khác.

---

### 🧑‍💻 Dương Dương — Retrieval & Lexical Search

**Phạm vi**: Thu thập tin tức, BM25 Lexical Search, Reranking, PageIndex Fallback

#### Files phụ trách (KHÔNG ai được chỉnh sửa song song):
- `src/task2_crawl_news.py` — Crawl tin tức
- `src/task6_lexical_search.py` — BM25 search
- `src/task7_reranking.py` — RRF reranking
- `src/task8_pageindex_vectorless.py` — PageIndex fallback
- `data/landing/news/` — Thư mục chứa bài crawl

#### Nhiệm vụ cụ thể:
1. **Crawl tin tức (Task 2)**:
   - Crawl tối thiểu **5 bài** vào `data/landing/news/`
   - Mỗi JSON có đủ: `url`, `title`, `date_crawled`, `content_markdown`
2. **BM25 Lexical Search (Task 6)**:
   - Implement `lexical_search(query, top_k=10) -> list[SearchResult]`
   - Output đúng schema `SearchResult`, `retrieval_method="bm25"`, sorted by score giảm dần
   - BM25 chạy trên **cùng corpus chunks** với dense search (Task 4)
3. **RRF Reranking (Task 7)**:
   - Implement `rerank_rrf(ranked_lists, top_k=5, k=60) -> list[SearchResult]`
   - Dùng công thức `sum(1 / (k + rank))`, rank bắt đầu từ 1
   - Chỉ fuse **một lần**
4. **PageIndex Fallback (Task 8)**:
   - Implement `pageindex_search(query, top_k=5) -> list[SearchResult]`
   - Trả `retrieval_method="pageindex"`, không crash UI khi provider lỗi
5. **Hoàn thiện `group_project/individual/INDIVIDUAL_REPORT.md`** của bản thân

> ⚠️ **Lưu ý tránh conflict**: Task 7 (RRF) được Dương implement, nhưng **Nhân** mới là người gọi nó trong `task9_retrieval_pipeline.py`. Dương chỉ expose function `rerank_rrf()` — không sửa Task 9.

---

## 🗂️ Ownership Map (tóm tắt)

| File / Thư mục | Chủ sở hữu |
|---------------|------------|
| `src/task1_collect_legal_docs.py` | Quang |
| `src/task2_crawl_news.py` | Dương |
| `src/task3_convert_markdown.py` | Quang |
| `src/task4_chunking_indexing.py` | Quang |
| `src/task5_semantic_search.py` | Quang |
| `src/task6_lexical_search.py` | Dương |
| `src/task7_reranking.py` | Dương |
| `src/task8_pageindex_vectorless.py` | Dương |
| `src/task9_retrieval_pipeline.py` | Nhân |
| `src/task10_generation.py` | Nhân |
| `src/contracts.py` | Nhân |
| `app.py` | Nhân |
| `pyproject.toml` | Nhân |
| `data/landing/legal/` | Quang |
| `data/landing/news/` | Dương |
| `tests/` | Nhân |
| `group_project/evaluation/RESULT.md` | Nhân |
| `group_project/individual/` (mỗi file riêng) | Từng thành viên |

---

## 🌿 Git Workflow — Tránh conflict

### Branch strategy

```
main
├── feature/quang-data-indexing     ← Quang làm ở đây
├── feature/duong-retrieval-search   ← Dương làm ở đây
└── feature/nhan-pipeline-eval      ← Nhân làm ở đây
```

### Quy tắc bắt buộc

1. **Mỗi người làm trên branch riêng** — không commit thẳng lên `main`.
2. **Chỉ sửa file trong phạm vi của mình** (xem Ownership Map ở trên).
3. Trước khi tạo PR, **rebase/merge từ `main`** để cập nhật thay đổi mới nhất.
4. **Nhân review và merge PR** sau khi code pass tests.
5. Không commit `.env`, API key, file cache, hoặc thư mục `__pycache__/`.
6. Commit message theo format: `[task-X] mô tả ngắn gọn` (ví dụ: `[task4] implement chunking và indexing`).

### Timeline đề xuất (3 giờ)

| Thời gian | Nhân | Quang | Dương |
|-----------|------|-------|-------|
| 0–10 phút | Setup môi trường, review contracts | Setup môi trường | Setup môi trường |
| 10–35 phút | Review & define schema trong `contracts.py` | Task 1: thu thập legal docs | Task 2: crawl news |
| 35–65 phút | Bắt đầu Task 9 skeleton | Task 3 + Task 4 | Task 6 + Task 7 |
| 65–95 phút | Task 9 hoàn chỉnh | Task 5: semantic search | Task 8: pageindex fallback |
| 95–125 phút | Task 10: generation + citation | Merge PR vào main | Task 8: pageindex fallback |
| 125–155 phút | `app.py` UI + golden dataset + evaluation | Individual report | Individual report |
| 155–180 phút | RESULT.md + tests + merge tất cả PR, demo | Individual report | Individual report |
