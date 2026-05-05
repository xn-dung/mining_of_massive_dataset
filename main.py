import os
import pandas as pd
import torch
import zipfile
from sentence_transformers import SentenceTransformer, util

# ==========================================
# CẤU HÌNH
# ==========================================
DATA_DIR = 'Cranfield' # Thư mục chứa 1400 file .txt và test.csv
TOP_K = 7 # Số lượng tài liệu liên quan trả về cho mỗi query. Bạn có thể tinh chỉnh số này.

# ==========================================
# 1. ĐỌC DỮ LIỆU TÀI LIỆU (DOCUMENTS)
# ==========================================
print("Đang đọc dữ liệu tài liệu...")
docs = []
doc_ids = []

for i in range(1, 1401):
    file_path = os.path.join(DATA_DIR, f"{i}.txt")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Xóa các ký tự xuống dòng để tạo thành một đoạn văn bản liền mạch
            content = f.read().replace('\n', ' ').strip()
            docs.append(content)
            doc_ids.append(str(i))
            
print(f"Đã tải {len(docs)} tài liệu.")

# ==========================================
# 2. ĐỌC DỮ LIỆU TRUY VẤN (QUERIES)
# ==========================================
print("Đang đọc dữ liệu truy vấn...")
test_csv_path = os.path.join(DATA_DIR, 'test.csv')
queries_df = pd.read_csv(test_csv_path)

queries = queries_df['query'].tolist()
query_ids = queries_df['query_id'].tolist()
print(f"Đã tải {len(queries)} truy vấn.")

# ==========================================
# 3. KHỞI TẠO MÔ HÌNH SENTENCE-BERT
# ==========================================
# Sử dụng mô hình được tối ưu hóa cho Information Retrieval / QA
print("Đang tải mô hình Sentence-BERT...")
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# ==========================================
# 4. TẠO EMBEDDINGS & TÍNH ĐỘ TƯƠNG ĐỒNG
# ==========================================
print("Đang mã hóa văn bản thành Vector Embeddings...")
# Mã hóa Documents (thường mất một chút thời gian tùy vào phần cứng)
doc_embeddings = model.encode(docs, convert_to_tensor=True, show_progress_bar=True)

# Mã hóa Queries
query_embeddings = model.encode(queries, convert_to_tensor=True, show_progress_bar=True)

# Tính toán ma trận Cosine Similarity giữa tất cả queries và documents
print("Đang tính toán độ tương đồng (Cosine Similarity)...")
cosine_scores = util.cos_sim(query_embeddings, doc_embeddings)

# ==========================================
# 5. LỌC KẾT QUẢ & TẠO FILE SUBMISSION
# ==========================================
print("Đang tổng hợp kết quả...")
results = []

for i in range(len(queries)):
    scores = cosine_scores[i]
    
    # Lấy ra Top K tài liệu có điểm tương đồng cao nhất
    # Lưu ý: Cranfield có số lượng relevant docs trung bình mỗi query khoảng 7-8.
    top_results = torch.topk(scores, k=TOP_K)
    
    # Lấy doc_id tương ứng với các index tìm được
    retrieved_doc_ids = [doc_ids[idx] for idx in top_results.indices]
    
    # Nối các doc_id bằng dấu cách theo đúng format
    relevant_docs_str = " ".join(retrieved_doc_ids)
    
    results.append({
        'query_id': query_ids[i],
        'query': queries[i],
        'relevant_docs': relevant_docs_str
    })

# Lưu ra file submission.csv
submission_df = pd.DataFrame(results)
submission_df.to_csv('submission.csv', index=False)

# Nén thành file submission.zip
with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write('submission.csv')

print("Hoàn tất! Đã tạo file submission.zip sẵn sàng để nộp bài.")