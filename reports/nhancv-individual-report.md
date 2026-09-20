# Individual contribution report

## Thong tin

- Ho va ten: Chu Van Nhan
- Ma hoc vien: K4-L3A-NhanCV
- Nhom: K4-L3A
- Repository/branch: https://github.com/VanNahAI/K4-L3A-RAG-Pipeline-NhanCV- / main

## Phan viec da thuc hien

| Module/deliverable | Viec toi truc tiep lam | File/commit/PR | Trang thai |
|---|---|---|---|
| Phan chia cong viec va Quan tri repo | Thiet ke teammate.md, phan quyen file tranh conflict, review va merge code toan nhom | teammate.md, commits 3c4e03c, fbf6e73, c6e9115 | Done |
| Module Contracts | Chuan hoa schema Document, Chunk, SearchResult, GenerationResult; thiet ke validators | src/contracts.py | Done |
| Retrieval Pipeline (Task 9) | Ket hop Dense + BM25, goi RRF fusion duy nhat 1 lan, co che cosine threshold fallback sang PageIndex | src/task9_retrieval_pipeline.py | Done |
| Generation va Citation (Task 10) | Lost-in-the-middle context reorder, LLM dispatcher (OpenAI, Gemini, Claude), Safe Refusal | src/task10_generation.py | Done |
| Chatbot UI | Streamlit app ket noi end-to-end pipeline, hien thi cau tra loi kem citation expander va diem so | app.py | Done |
| Evaluation va Testing | Xay dung 15 ca hoi-dap golden dataset, hoan thanh bao cao RESULT.md A/B benchmark, pass 20/20 pytest | group_project/evaluation/golden_dataset.json, RESULT.md | Done |

## Quyet dinh ky thuat quan trong

1. **Quyet dinh:** Su dung diem Cosine Similarity goc cua Dense Retrieval de quyet dinh Fallback thay vi dung diem RRF.
   **Ly do/evidence:** Diem RRF chi la tong nghich dao thu tu xep hang (thang diem phu thuoc so luong danh sach gop va hang so k), khong phan anh do tuong dong ngu nghia thuc te. Neu query hoan toan ngoai mien (out-of-domain), ca BM25 va Dense deu tra ve cac chunk xep hang 1-2 voi RRF score cao du khong lien quan. Viec dung nguong cosine goc (<0.30) kich hoat PageIndex fallback hoac Safe Refusal chinh xac 100% tren 2 cau hoi test ngoai mien.
   **Trade-off:** Can bao luu cosine score goc cua dense search xuyen suot pipeline ma khong de cac buoc chuan hoa sau lam mat.

2. **Quyet dinh:** Ap dung chien luoc Context Reordering (Lost-in-the-Middle) cho generation.
   **Ly do/evidence:** Theo nghien cuu cua Liu et al., LLM thuong chu y tot nhat vao cac doan van o dau va cuoi prompt, de bo qua bang chung o giua. Bang cach chia tach index chan (front) va le dao nguoc (back), tai lieu top-1 va top-2 nam o 2 dau context, giup diem Faithfulness tang tu 0.88 len 0.96.
   **Trade-off:** Can dam bao thuat toan sap xep la non-mutating de khong lam xao tron list goc, giu nguyen chunk_index va mapping citation ve Document N.

## Kiem thu va ket qua

- Test hoac query toi da dung: Bo 20 test cases tu dong gom test_contracts.py va test_acceptance.py (pytest -v). Cac query test thuc te: Quy che bo xet tuyen som cua Bo GD-DT, Diem chuan cao nhat HUST 2025, va query out-of-domain: Lich trinh bieu dien concert tai SVD My Dinh.
- Ket qua truoc/sau neu co: Truoc khi toi uu, cau hoi ngoai mien bia thong tin tu context gan nhat; sau khi ap dung threshold 0.30 + safe refusal, he thong tu choi chuan xac: Toi khong the xac minh thong tin nay tu nguon hien co.
- Loi da phat hien va cach xu ly: Phat hien rang buoc requires-python lam moi truong Python 3.14 khong cai duoc rank-bm25 va langchain-text-splitters; da noi long va cai dat thanh cong, dua so test pass tu 13/20 len 20/20 (100%).

## Dieu con han che

- Mot han che cu the cua phan toi lam: Do tre cua PageIndex fallback phu thuoc vao polling HTTP request tu dich vu ben thu ba khi xay ra out-of-domain query.
- Neu co them thoi gian, thay doi dau tien toi se thuc hien: Bo sung co che cache bat dong bo (Async Redis hoac SQLite cache) cho cac ket qua truy xuat tuong tu de giam latency truy van xuong duoi 100ms.

## Xac nhan dong gop

Toi xac nhan noi dung tren phan anh dung phan viec cua minh va co the giai thich hoac chay lai trong buoi demo.

- Ngay: 2026-09-20
- Ten thanh vien: Chu Van Nhan
