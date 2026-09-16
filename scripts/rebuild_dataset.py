import pandas as pd
import os
import re
from urllib.parse import unquote

# Đường dẫn
PATH_CSIC = '/home/dxthanh/hin_web_vulne/data/labeled_csic_database.csv'
PATH_XSS_SOURCE = '/home/dxthanh/hin_web_vulne/data/XSS_dataset.csv'
OUTPUT_BALANCED = '/home/dxthanh/hin_web_vulne/data/augmented_web_attack.csv'

def clean_text(text):
    if pd.isna(text) or str(text).lower() == 'nan': return None
    text = unquote(str(text)) # Giải mã URL ngay từ đầu
    if len(text.strip()) < 2: return None
    return text

def rebuild_data():
    print("--- BẮT ĐẦU QUY TRÌNH TÁI THIẾT DỮ LIỆU SẠCH ---")
    
    # 1. Đọc và dọn dẹp CSIC gốc
    df_csic = pd.read_csv(PATH_CSIC)
    df_csic['content'] = df_csic['content'].apply(clean_text)
    df_csic = df_csic.dropna(subset=['content']).drop_duplicates(subset=['content'])
    
    # Tách lớp sau khi dọn
    df_normal_clean = df_csic[df_csic['attack_type'] == 0]
    df_sqli_clean = df_csic[df_csic['attack_type'] == 1]
    
    print(f"[*] Quỹ dữ liệu sạch: Normal ({len(df_normal_clean)}), SQLi ({len(df_sqli_clean)})")

    # 2. Xử lý XSS Source
    df_xss_source = pd.read_csv(PATH_XSS_SOURCE)
    payload_col = 'Sentence' if 'Sentence' in df_xss_source.columns else df_xss_source.columns[4]
    xss_payloads = df_xss_source[df_xss_source['Label'] == 1][payload_col].dropna().unique().tolist()
    print(f"[*] Quỹ XSS sạch: {len(xss_payloads)} mẫu")

    # 3. Xác định mục tiêu cân bằng (Lấy theo lớp đông nhất hoặc con số 10,000)
    target_size = max(len(df_normal_clean) // 2, len(df_sqli_clean), 10000)
    print(f"[*] Mục tiêu cân bằng mỗi lớp: {target_size}")

    # --- TẠO LỚP 0: BENIGN ---
    # Lấy 1/2 Normal làm Benign
    df_benign = df_normal_clean.sample(n=len(df_normal_clean)//2, random_state=42)
    # Nhân bản cho đủ target_size
    df_benign_final = pd.concat([df_benign] * (target_size // len(df_benign) + 1)).iloc[:target_size]

    # --- TẠO LỚP 1: SQLi ---
    # Nhân bản SQLi gốc đã dọn sạch cho đủ target_size
    df_sqli_final = pd.concat([df_sqli_clean] * (target_size // len(df_sqli_clean) + 1)).iloc[:target_size]

    # --- TẠO LỚP 2: XSS (Đầu độc trên khung sạch) ---
    # Lấy 1/2 Normal còn lại làm khung
    df_frames = df_normal_clean.drop(df_benign.index)
    poisoned_records = []
    for i in range(target_size):
        # Lấy template từ khung sạch (dùng modulo để lặp lại khung nếu cần)
        template = df_frames.iloc[i % len(df_frames)].to_dict()
        template['content'] = xss_payloads[i % len(xss_payloads)]
        template['attack_type'] = 2
        poisoned_records.append(template)
    df_xss_final = pd.DataFrame(poisoned_records)

    # 4. Hợp nhất
    final_df = pd.concat([df_benign_final, df_sqli_final, df_xss_final])
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_df.to_csv(OUTPUT_BALANCED, index=False)
    print(f"\n[V] THÀNH CÔNG! File mới tại: {OUTPUT_BALANCED}")
    print(final_df['attack_type'].value_counts())

if __name__ == "__main__":
    rebuild_data()
