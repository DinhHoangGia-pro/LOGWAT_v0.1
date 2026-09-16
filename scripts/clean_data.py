import pandas as pd
import os

path = "/home/dxthanh/hin_web_vulne/data/augmented_web_attack.csv"
if not os.path.exists(path):
    print(f"[!] Không tìm thấy file: {path}")
    exit()

df = pd.read_csv(path)
print(f"[*] Tổng số dòng ban đầu: {len(df)}")

# 1. Xóa các dòng có cột content bị rỗng (NaN)
df = df.dropna(subset=['content'])

# 2. Xóa các dòng mà content chỉ chứa chuỗi 'nan' hoặc quá ngắn (< 2 ký tự)
df = df[df['content'].astype(str).str.lower() != 'nan']
df = df[df['content'].astype(str).str.len() > 1]

# 3. Xóa trùng lặp để tránh model học vẹt
df = df.drop_duplicates(subset=['content'])

print(f"[*] Số dòng sau khi làm sạch: {len(df)}")
# Kiểm tra phân phối lớp
print(df['attack_type'].value_counts())

df.to_csv(path, index=False)
print("[V] Đã dọn dẹp xong dữ liệu 'bẩn'!")
