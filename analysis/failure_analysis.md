# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Phạm Thị Thùy Linh  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.00 | **0.7083** | +0.7083 |
| Answer Relevancy | 0.00 | **0.6341** | +0.6341 |
| Context Precision | 0.00 | **0.9542** | +0.9542 |
| Context Recall | 0.00 | **0.7667** | +0.7667 |

> Naive baseline chạy không có OPENAI_API_KEY nên scores = 0. Production pipeline đạt 3/4 metrics ≥ 0.70.

---

## Bottom-5 Failures

### #1 — "Bao lâu phải đổi mật khẩu một lần?"
- **Worst metric:** faithfulness = 0.0 (avg_score = 0.3958)
- **Diagnosis:** LLM hallucinating — câu trả lời chứa thông tin không có trong context
- **Error Tree:**
  - Output sai? → **Có** (LLM thêm thông tin ngoài context)
  - Context retrieved đúng? → Có (context_precision = 1.0)
  - Chunking OK? → Có (structure-aware giữ được đoạn policy)
  - Root cause: **LLM tự suy luận thêm** khi chunk có policy v1 (90 ngày) và v2 (120 ngày) — không biết version nào là hiện hành
- **Suggested fix:** Thêm metadata `version` + `effective_date` vào chunk; prompt: *"Chỉ dùng thông tin trong context, ưu tiên version mới nhất"*; lower temperature (0.0)

---

### #2 — "Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?"
- **Worst metric:** faithfulness = 0.0 (avg_score = 0.4167)
- **Diagnosis:** LLM hallucinating — câu trả lời mang tính tư vấn chung, không dựa trên policy cụ thể
- **Error Tree:**
  - Output sai? → **Có** (LLM thêm lời khuyên generic về cybersecurity)
  - Context retrieved đúng? → Một phần (context_recall thấp — thiếu chunk IT security)
  - Query rewrite OK? → Không tốt — "malware" là từ tiếng Anh, BM25 không segment được
  - Root cause: **Vocabulary gap** — corpus dùng "phần mềm độc hại" nhưng query dùng "malware"
- **Suggested fix:** HyDE (Hypothetical Document Embedding) hoặc query expansion; enrichment thêm synonym tiếng Anh vào metadata

---

### #3 — "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"
- **Worst metric:** answer_relevancy = 0.0 (avg_score = 0.4583)
- **Diagnosis:** Answer không match question — câu hỏi multi-hop yêu cầu kết hợp thông tin từ 2 tài liệu khác nhau (leave policy + salary policy)
- **Error Tree:**
  - Output đúng format? → Không (chỉ trả lời 1 phần)
  - Context đủ? → Không — context chỉ chứa thông tin nghỉ phép, thiếu bảng lương Senior
  - Retrieval OK? → Không — hybrid search trả về top results từ cùng 1 domain
  - Root cause: **Multi-hop retrieval failure** — RRF không có cơ chế tổng hợp từ nhiều document domain khác nhau
- **Suggested fix:** Multi-hop retrieval: chạy 2 queries riêng ("nghỉ phép thâm niên" và "lương Senior"), sau đó merge context trước khi gọi LLM

---

### #4 — "Bảo hiểm sức khỏe PVI có hạn mức bao nhiêu cho nhân viên?"
- **Worst metric:** faithfulness = 0.0 (avg_score = 0.4665)
- **Diagnosis:** LLM hallucinating — đưa ra con số hạn mức không có trong corpus
- **Error Tree:**
  - Output sai? → **Có** (LLM bịa số tiền cụ thể)
  - Context có thông tin không? → Không rõ ràng — chunk về PVI chỉ đề cập tên gói, không có số tiền hạn mức cụ thể
  - Chunking OK? → Có thể là vấn đề — bảng số liệu trong markdown bị cắt ngang
  - Root cause: **Thiếu thông tin trong corpus** + LLM fill-in-the-blank thay vì nói "Không tìm thấy"
- **Suggested fix:** Thêm fallback prompt: *"Nếu không có thông tin cụ thể, trả lời 'Thông tin không có trong tài liệu'"*; kiểm tra lại chunking cho file bao_hiem.md

---

### #5 — "Nhân viên thử việc có được nghỉ phép năm không?"
- **Worst metric:** faithfulness = 0.0 (avg_score = 0.5)
- **Diagnosis:** LLM hallucinating — câu hỏi yêu cầu negation reasoning (thử việc KHÔNG được nghỉ phép)
- **Error Tree:**
  - Output đúng? → Không (LLM trả lời "có" hoặc mơ hồ)
  - Context chứa thông tin negation không? → Có nhưng LLM không extract được
  - Root cause: **Negation comprehension failure** — LLM gặp khó khăn với câu hỏi dạng "X có được Y không?" khi answer là "không"
- **Suggested fix:** Fine-tune prompt với chain-of-thought: *"Đọc kỹ điều kiện áp dụng trước khi trả lời"*; thêm ví dụ negation vào few-shot prompt

---

## Case Study — Phân tích sâu: Multi-hop Failure

**Question:** "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"

**Error Tree walkthrough:**
1. Output đúng? → **Không** — answer_relevancy = 0.0
2. Context đầy đủ? → **Không** — chỉ có thông tin về nghỉ phép, thiếu salary range Senior
3. Query rewrite OK? → **Không** — 1 query không thể cover 2 domain khác nhau
4. Fix ở bước: **Retrieval** — cần decompose query thành sub-queries trước khi search

**Nếu có thêm 1 giờ, sẽ optimize:**
- Implement query decomposition: dùng LLM tách multi-hop question thành 2–3 sub-questions
- Chạy hybrid search cho mỗi sub-question, merge contexts
- Kỳ vọng tăng answer_relevancy từ 0.63 lên 0.80+
