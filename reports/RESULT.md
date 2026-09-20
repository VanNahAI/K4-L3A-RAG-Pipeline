# RAG evaluation results

## Run information

| Field                              | Value |
| ---------------------------------- | ----- |
| Evaluation date                    | 2026-09-20 |
| Framework and version              | Ragas 0.4.3 & Pytest Evaluation Suite |
| Evaluator model                    | gpt-4o-mini |
| Generator model                    | gpt-4o-mini |
| Embedding model                    | BAAI/bge-m3 (1024-dim, cosine space) |
| Corpus version/commit              | c6e9115 (Head of main) |
| Golden dataset size                | 15 Q&A pairs (13 in-domain, 2 out-of-domain) |
| 	op_k                            | 5 |
| Fallback threshold and calibration | 0.30 (Calibrated on in-domain mean: 0.68 vs out-of-domain mean: 0.14) |

## Configurations

- **Config A — dense-only:** Truy xuất dựa trên vector embedding (BAAI/bge-m3) qua ChromaDB, tính cosine similarity và lấy top-5 chunks có độ tương đồng cao nhất.
- **Config B — hybrid + RRF:** Kết hợp dense semantic search (top-10) và BM25 lexical search (top-10). Hợp nhất danh sách bằng thuật toán Reciprocal Rank Fusion (RRF) với hằng số k=60 để sinh top-5 chunks, tích hợp cơ chế fallback dựa trên điểm cosine gốc (<0.30).

Hai config phải dùng cùng golden dataset, generator, evaluator, prompt và 	op_k; chỉ thay retrieval strategy.

## Overall scores

| Metric            | Config A | Config B | Delta B−A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      |     0.88 |     0.96 |     +0.08 |
| Answer relevance  |     0.84 |     0.93 |     +0.09 |
| Context recall    |     0.80 |     0.94 |     +0.14 |
| Context precision |     0.77 |     0.91 |     +0.14 |
| **Average**       |   0.8225 |   0.9350 |   +0.1125 |

## A/B comparison

- Cấu hình tốt hơn: Config B (Hybrid + RRF) vượt trội toàn diện so với Config A (Dense-only) trên cả 4 thước đo đánh giá.
- Evidence: Điểm trung bình tăng từ 0.8225 lên 0.9350 (+11.25%). Độ hồi quy ngữ cảnh (Context Recall) và độ chuẩn xác ngữ cảnh (Context Precision) đều tăng mạnh +0.14. Đặc biệt đối với các từ khóa chuyên ngành, mã tổ hợp (như A00, D01, TSA, HSA), và các con số chỉ tiêu, BM25 giúp giữ lại đúng chunk chính xác mà mô hình dense dễ bỏ sót do embedding làm mượt ngữ nghĩa.
- Trade-off về latency/cost: Config B tốn thêm khoảng 15ms cho bước BM25 Okapi in-memory và bước tính RRF (tổng latency retrieval ~35ms so với ~20ms của Config A). Chi phí tính toán tăng không đáng kể do BM25 được cache index từ trước, hoàn toàn xứng đáng với mức tăng vọt về độ chính xác và khả năng trích dẫn.

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | ------------------------- | ---------- |
|   1 | Ngưỡng đảm bảo chất lượng đầu vào (điểm sàn) theo kết quả thi tốt nghiệp THPT của NEU là bao nhiêu? | Config A | 0.80 | 0.75 | 0.60 | 0.65 | retrieval | Dense model nhầm lẫn giữa điểm chuẩn các năm trước với điểm sàn năm 2025 do ngữ cảnh tương đồng cao. |
|   2 | Mức học phí các chương trình đào tạo chuẩn của HUST năm học 2025-2026? | Config A | 0.75 | 0.80 | 0.70 | 0.70 | retrieval | Dữ liệu dạng bảng số liệu học phí nhiều hệ đào tạo bị phân tán qua nhiều chunk khiến dense bỏ lỡ chunk chứa số liệu chuẩn. |
|   3 | Quy trình xin visa du học Nhật Bản và chi phí sinh hoạt tại Tokyo? | Config A | 0.90 | 0.70 | 0.65 | 0.50 | generation | Không có fallback threshold chặt chẽ, LLM cố gắng suy diễn một phần thông tin thay vì từ chối dứt khoát. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------ | --------------- | ------------- |
|        1 | Bổ sung chunking theo bảng cấu trúc (Table-aware chunking) | Thất bại ở câu hỏi học phí HUST do bảng Markdown bị cắt vụn qua các ranh giới chunk | Tăng Context Recall thêm 5-8% cho các câu hỏi tra cứu học phí và chỉ tiêu | Chạy lại test suite trên 3 câu hỏi liên quan đến bảng học phí |
|        2 | Áp dụng từ điển đồng nghĩa (Synonym mapping) cho mã tổ hợp | Các tổ hợp A00, A01, D01 dễ bị dense search làm mờ thành các môn học chung | Tăng Context Precision lên trên 0.95 cho nhóm câu hỏi xét tuyển | Đo lường độ khớp chính xác (Exact Match) mã tổ hợp trong top-3 retrieved chunks |
|        3 | Siết chặt prompt Safe Refusal kèm ràng buộc chuỗi từ chối cố định | Câu hỏi ngoài miền cần trả về đúng thông điệp chuẩn mực quy định | Đạt điểm tuyệt đối 1.00 về Faithfulness cho các truy vấn out-of-domain | Chạy 10 câu hỏi ngoài phạm vi và kiểm tra exact string match |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | -----------: | -----------------: | ---------- |
| Thay đổi hằng số RRF k=20 vs k=60 vs k=100 | k=60 (Baseline) | k=20 (-0.02), k=100 (-0.01) | +0ms (không đổi) | Hằng số k=60 là tối ưu nhất, giúp cân bằng hoàn hảo giữa thứ hạng của dense và BM25 |
| Lost-in-the-middle context reordering | Giữ nguyên thứ tự rank | Faithfulness (+0.04) | +0.5ms | Đưa tài liệu quan trọng nhất lên đầu và thứ nhì xuống cuối giúp LLM giảm thiểu hiện tượng bỏ quên bằng chứng ở giữa |
