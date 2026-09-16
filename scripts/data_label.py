import pandas as pd
import re
import os
import logging

# --- Cấu hình Đường dẫn ---
INPUT_FILE = '/home/dxthanh/hin_web_vulne/data/csic_database.csv'
OUTPUT_FILE = '/home/dxthanh/hin_web_vulne/data/labeled_csic_database.csv'
LOG_DIR = '../logs'

# Khởi tạo thư mục và logging
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "labeling_process.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def web_attack_classifier(row):
    """
    Phân loại dựa trên đặc trưng hành vi Token Web.
    Nhãn: 0: Normal, 1: SQLi, 2: XSS, 3: CSRF/Other Anomalies
    """
    # Bước 1: Kiểm tra nhãn gốc của CSIC 2010 (Thường là Normal/Anomalous)
    # Nếu là Normal, ta giữ nguyên nhãn 0 để tránh nhiễu
    if str(row.get('classification', '')).lower() == 'normal' or row.get('classification') == 0:
        return 0

    # Bước 2: Trích xuất nội dung từ URL và Body (content)
    # Tấn công Web thường nằm ở tham số URL hoặc payload trong Body
    target_data = (str(row) + " " + str(row['content'])).lower()

    # --- Ưu tiên XSS trước vì Regex XSS thường đặc thù hơn ---
    xss_patterns = [
        r"<script.*?>", r"alert\(", r"onerror=", r"onload=", 
        r"javascript:", r"document\.cookie", r"<img.*?src=", 
        r"onmouseover=", r"eval\(", r"window\.location"
    ]
    if any(re.search(p, target_data) for p in xss_patterns):
        return 2

    # --- SQL Injection Patterns ---
    sqli_patterns = [
        r"union.*select", r"select.*from", r"insert.*into", r"drop.*table",
        r"or\s+\d+=\d+", r"sleep\(", r"benchmark\(", r"information_schema",
        r"admin'--", r"order\s+by"
    ]
    if any(re.search(p, target_data) for p in sqli_patterns):
        return 1

    # --- Các loại khác (Directory Traversal, CRLF...) ---
    other_patterns = [r"\.\./\.\./", r"etc/passwd", r"%0d%0a", r"boot\.ini"]
    if any(re.search(p, target_data) for p in other_patterns):
        return 3

    return 1 # Mặc định gán nhãn 1 (SQLi) cho các Anomalous còn lại vì đây là lớp đa số

def main():
    try:
        logging.info("Bắt đầu đọc file: " + INPUT_FILE)
        df = pd.read_csv(INPUT_FILE)
        
        # Chuẩn hóa tên cột
        df.columns = [c.replace('lenght', 'length') for c in df.columns]

        logging.info("Đang tiến hành gán nhãn đa lớp...")
        df['attack_type'] = df.apply(web_attack_classifier, axis=1)

        df.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"Ghi file thành công: {OUTPUT_FILE}")
        print(f"Hoàn thành! Kết quả tại: {OUTPUT_FILE}")

    except Exception as e:
        logging.error(f"Lỗi: {str(e)}")
        print(f"Lỗi thực thi. Kiểm tra log tại {LOG_DIR}")

if __name__ == "__main__":
    main()
