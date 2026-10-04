# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Phạm Thị Thùy Linh  
**MSSV:** 2A202602909  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 05/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Dùng `all-MiniLM-L6-v2` encode từng câu, cosine similarity < threshold (0.85) → tách chunk mới. Tạo ra ~40% ít chunks hơn basic chunking nhờ gộp các câu cùng chủ đề — giảm context bị cắt giữa ý. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Parent (2048 chars) chứa context đầy đủ; children (256 chars) dùng để retrieve chính xác. Children có `parent_id` để khi cần thiết có thể expand lên parent — đây là pattern production tốt nhất cho balance giữa precision và context. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | `re.split(r'(^#{1,3}\s+.+$)', ...)` tách đúng theo markdown headers. Mỗi chunk giữ `section` metadata — rất hữu ích khi filter theo phòng ban hoặc chủ đề (VD: chỉ tìm trong section "Nghỉ phép"). |
| BM25 + Dense fusion (Hybrid) | M2 | `reciprocal_rank_fusion()` | RRF score = Σ 1/(60 + rank + 1). BM25 giỏi với exact match ("nghỉ_phép", số liệu); Dense giỏi với semantic similarity. RRF kết hợp cả hai mà không cần tune trọng số — context_precision đạt 0.954 nhờ hybrid. |
| Vietnamese segmentation | M2 | `segment_vietnamese()` | underthesea nối từ ghép bằng `_` (VD: "nghỉ_phép") → phải replace `_`→` ` trước khi tokenize. Nếu không, BM25 không khớp query "nghỉ phép" (2 tokens) với corpus "nghỉ_phép" (1 token). |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `BAAI/bge-reranker-v2-m3` score từng pair (query, doc) — chính xác hơn bi-encoder nhưng O(n) thay vì O(1). Top-20 → top-3. Latency ~2s/query; precision cải thiện rõ rệt so với chỉ dùng dense search. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Faithfulness đo LLM có hallucinate không; Answer Relevancy đo câu trả lời có liên quan câu hỏi không; Context Precision đo chunks retrieved có đúng không; Context Recall đo có đủ chunks không. Metric thấp nhất: Answer Relevancy (0.63) — do multi-hop questions. |
| Contextual embeddings (M5) | M5 | `_enrich_single_call()` | 1 API call/chunk trả về summary + 3 hypothesis questions + context prefix + metadata. Contextual prepend giảm retrieval failure theo Anthropic benchmark (-49%). Combined mode tiết kiệm 4x API calls so với gọi riêng lẻ. |

---

## Phần 2: Khó khăn & Cách giải quyết

### Lỗi 1: `No module named 'sentence_transformers'`
- **Error:** `ModuleNotFoundError: No module named 'sentence_transformers'` khi chạy test M1
- **Nguyên nhân:** `sentence-transformers` chưa được cài mặc dù có trong `requirements.txt`
- **Debug:** Chạy `pip install sentence-transformers` riêng; phát hiện do pip install lần đầu bị timeout không cài hết
- **Fix:** Cài lại từng package riêng lẻ thay vì `pip install -r requirements.txt` một lần

### Lỗi 2: Conflict numpy/scipy/ragas versions
- **Error:** `AttributeError: module 'numpy' has no attribute 'long'` khi chạy pipeline
- **Nguyên nhân:** `ragas 0.1.x` kéo theo `scipy` cũ không tương thích `numpy 2.x`; `ragas 0.2.x` lại kéo `langchain 0.2.x` yêu cầu `numpy < 2.0`
- **Debug:** Đọc traceback, trace ngược từ `scipy.sparse._sputils` → `numpy.long` deprecated
- **Fix:** Upgrade lên `ragas 0.2.x` + `numpy 2.x` + `scipy 1.18+`. Cập nhật `evaluate_ragas()` để tương thích API mới (class-based metrics, đổi tên cột `question`→`user_input`)

### Lỗi 3: RAGAS `'question'` KeyError
- **Error:** `RAGAS evaluation failed: 'question'` khi parse DataFrame
- **Nguyên nhân:** ragas 0.2 đổi tên các cột output: `question`→`user_input`, `answer`→`response`, `ground_truth`→`reference`
- **Debug:** Chạy test đơn lẻ với 1 câu hỏi, in `list(df.columns)` để xem tên thực tế
- **Fix:** Map dynamic cột theo version ragas được detect tự động trong code

### Lỗi 4: Enrichment API trả về empty JSON (`Expecting value: line 1 column 1`)
- **Nguyên nhân:** `gpt-4o-mini` đôi khi trả về response có markdown code fence (```json...```) thay vì raw JSON — `json.loads()` fail
- **Fix (partial):** Một số chunk bị skip và dùng fallback (raw text). Nếu có thêm thời gian sẽ thêm strip markdown fences trước khi parse

---

## Phần 3: Action Plan cho Project cá nhân

### Project: Chatbot hỏi đáp tài liệu nội bộ doanh nghiệp

#### 1. Hiện trạng
- **Pipeline hiện tại:** Basic chunking (500 chars) + Dense search only (OpenAI ada-002) + GPT-4o answer generation
- **Vấn đề đang gặp:**
  - Retrieval miss khi user hỏi bằng từ khóa chính xác (VD: mã sản phẩm, số điều khoản)
  - LLM hallucinate khi tài liệu có nhiều version (policy cũ/mới)
  - Không đánh giá được chất lượng pipeline một cách hệ thống

#### 2. Kế hoạch cải tiến

1. **Chunking strategy:** Dùng **Structure-aware** cho tài liệu có cấu trúc rõ (hợp đồng, policy) + **Hierarchical** cho tài liệu dài (báo cáo). Lý do: giữ được metadata section, không cắt giữa điều khoản.

2. **Search retrieval:** Chuyển sang **Hybrid Search (BM25 + Dense + RRF)**. Lý do: BM25 bắt được mã số, số điều khoản; Dense bắt được intent. Giữ underthesea segmentation cho tiếng Việt.

3. **Reranking:** Thêm `BAAI/bge-reranker-v2-m3` cho top-20→top-5. Chấp nhận latency +2s để tăng precision từ 0.7 lên 0.9+.

4. **Evaluation:** Tích hợp **RAGAS 4 metrics** chạy weekly trên golden test set 50 câu. Alert khi faithfulness < 0.70. Hiện tại không có đánh giá định kỳ nào.

5. **Enrichment:** Áp dụng **Contextual prepend** (Anthropic -49% retrieval failure) + **Auto metadata extraction** (category, version, effective_date). Dùng combined mode để tiết kiệm API cost.

#### 3. Timeline triển khai

- **Tuần 1:** Implement Structure-aware chunking + Hierarchical cho corpus hiện tại; re-index toàn bộ
- **Tuần 2:** Thêm BM25 index + RRF fusion; A/B test với pipeline cũ trên 20 câu hỏi thực tế
- **Tuần 3:** Tích hợp reranker; đo latency trước/sau; benchmark trên golden test set
- **Tuần 4:** Thêm RAGAS evaluation pipeline; viết script chạy weekly; setup alert
- **Tuần 5:** Enrichment với contextual prepend + metadata; đo improvement retrieval recall
