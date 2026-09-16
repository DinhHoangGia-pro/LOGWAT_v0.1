import pandas as pd
import os
import logging
import sys

# --- Cấu hình Đường dẫn tuyệt đối ---
PATH_CSIC = '/home/dxthanh/hin_web_vulne/data/labeled_csic_database.csv'
PATH_XSS_SOURCE = '/home/dxthanh/hin_web_vulne/data/XSS_dataset.csv'
OUTPUT_BALANCED = '/home/dxthanh/hin_web_vulne/data/augmented_web_attack.csv'
LOG_DIR = '/home/dxthanh/hin_web_vulne/logs'

# Đảm bảo thư mục log tồn tại
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Cấu hình Logging để Debug
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'augmentation_debug.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def augment_xss_data():
    try:
        logging.info("--- BẮT ĐẦU QUY TRÌNH ĐẦU ĐỘC PAYLOAD (14,000 SAMPLES) ---")
        
        # 1. Đọc dữ liệu
        print("Đang đọc dữ liệu nguồn và tính toán số lượng...")
        df_csic = pd.read_csv(PATH_CSIC)
        df_xss_source = pd.read_csv(PATH_XSS_SOURCE)
        
        # 2. Trích xuất mã độc XSS thực sự (Label = 1) 
        payload_col = 'Sentence' if 'Sentence' in df_xss_source.columns else df_xss_source.columns[4]
        malicious_xss_list = df_xss_source[df_xss_source['Label'] == 1][payload_col].astype(str).tolist()
        num_xss_payloads = len(malicious_xss_list)
        
        logging.info(f"Số lượng mã độc XSS khả dụng trong file nguồn: {num_xss_payloads}")

        # 3. Tách các lớp từ dữ liệu CSIC 2010
        df_normal_all = df_csic[df_csic['attack_type'] == 0]
        df_sqli_all = df_csic[df_csic['attack_type'] == 1]
        
        # Mục tiêu 14,000 cho mỗi lớp
        num_target = 14000
        
        # Kiểm tra điều kiện dữ liệu Normal (cần 14k làm Benign và 14k làm khung đầu độc = 28k)
        if len(df_normal_all) < num_target * 2:
            num_target = len(df_normal_all) // 2
            logging.warning(f"Dữ liệu Normal không đủ 28k. Hạ mục tiêu xuống: {num_target} mỗi loại.")

        # 4. Chuẩn bị các tập dữ liệu thành phần
        # Lấy 14k mẫu làm khung để đầu độc
        frames_to_poison = df_normal_all.sample(n=num_target, random_state=42).copy()
        # Loại bỏ các mẫu đã dùng làm khung khỏi tập Benign để tránh trùng lặp
        df_benign_final = df_normal_all.drop(frames_to_poison.index).sample(n=num_target, random_state=42)
        # Lấy 14k mẫu SQLi
        df_sqli_final = df_sqli_all.sample(n=num_target, random_state=42)

        # 5. Thực hiện "đầu độc" (Poisoning) với cơ chế Modulo để tránh IndexError
        poisoned_records = []
        
        logging.debug(f"Bắt đầu chèn {num_target} mẫu XSS vào khung request...")
        for i in range(num_target):
            # Lấy record normal làm template
            row = frames_to_poison.iloc[i].to_dict()
            # Sử dụng toán tử % để lặp lại payload XSS nếu i vượt quá số lượng payload có sẵn 
            row['content'] = malicious_xss_list[i % num_xss_payloads]
            row['attack_type'] = 2 # Gán nhãn 2 cho XSS
            poisoned_records.append(row)
            
            if (i+1) % 5000 == 0:
                logging.debug(f"Đã xử lý {i+1}/{num_target} mẫu XSS.")
        
        df_xss_augmented = pd.DataFrame(poisoned_records)

        # 6. Hợp nhất thành tập dữ liệu cân bằng (14k Normal + 14k SQLi + 14k XSS)
        final_df = pd.concat([
            df_benign_final,
            df_sqli_final,
            df_xss_augmented
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 7. Lưu kết quả
        final_df.to_csv(OUTPUT_BALANCED, index=False)
        
        msg = f"THÀNH CÔNG! Đã tạo tập dữ liệu: {len(final_df)} mẫu (14k mỗi loại).\n"
        msg += f"Phân bổ nhãn:\n{final_df['attack_type'].value_counts()}"
        print(msg)
        logging.info(msg)

    except Exception as e:
        error_msg = f"LỖI THỰC THI: {str(e)}"
        print(error_msg)
        logging.error(error_msg, exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    augment_xss_data()
