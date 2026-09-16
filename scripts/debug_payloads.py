import torch
import pickle
import os
import pandas as pd
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm, global_max_pool, JumpingKnowledge

# --- CONFIG ---
HIN_DIR = "/home/dxthanh/hin_web_vulne"
DATA_PATH = os.path.join(HIN_DIR, "data/web_graphs.pkl")
MODEL_PATH = os.path.join(HIN_DIR, "data/models_pretrained/best_web_gnn.pth")
ORIGINAL_CSV = os.path.join(HIN_DIR, "data/augmented_web_attack.csv")

# --- RE-DECLARE MODEL CLASS (Phải khớp với train_web_gnn.py) ---
class HeavyWebGNN(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=512, out_channels=3):
        super().__init__()
        self.preprocess = nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.ReLU(), nn.Dropout(0.2))
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4)
        self.bn3 = BatchNorm(hidden_dim)
        self.jk = JumpingKnowledge(mode='cat')
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, x, edge_index, batch):
        x = self.preprocess(x)
        x1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        x2 = F.elu(self.bn2(self.conv2(x1, edge_index)))
        x3 = F.elu(self.bn3(self.conv3(x2, edge_index)))
        x_jk = self.jk([x1, x2, x3])
        return self.classifier(global_max_pool(x_jk, batch))

def debug_analysis():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Kiểm tra File
    if not os.path.exists(ORIGINAL_CSV):
        print(f"[!] Không thấy file CSV tại {ORIGINAL_CSV}")
        return
    if not os.path.exists(DATA_PATH):
        print(f"[!] Không thấy file đồ thị tại {DATA_PATH}")
        return

    # 2. Load Data
    print("[*] Đang nạp dữ liệu...")
    df = pd.read_csv(ORIGINAL_CSV)
    with open(DATA_PATH, 'rb') as f:
        data_pkl = pickle.load(f)
    dataset = data_pkl['graphs']
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 3. Load Model
    model = HeavyWebGNN().to(device)
    if os.path.exists(MODEL_PATH):
        print(f"[*] Đang nạp trọng số từ {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("[!] CẢNH BÁO: Không tìm thấy file .pth! Model đang dùng trọng số ngẫu nhiên.")
        print("[!] Kết quả dưới đây có thể không chính xác cho đến khi bạn train thành công ít nhất 1 epoch có Acc cao hơn.")
    
    model.eval()

    missed_sqli = [] # Thực tế SQLi nhưng đoán Benign
    false_positives = [] # Thực tế Benign nhưng đoán SQLi

    print("[*] Đang quét toàn bộ tập dữ liệu để tìm lỗi...")
    
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1).item()
            actual = data.y.item()
            
            payload = df.iloc[i]['content']
            
            # Case 1: Missed SQLi (Actual 1, Pred 0)
            if actual == 1 and pred == 0 and len(missed_sqli) < 10:
                missed_sqli.append(payload)
            
            # Case 2: False Positive (Actual 0, Pred 1)
            if actual == 0 and pred == 1 and len(false_positives) < 10:
                false_positives.append(payload)
                
            if len(missed_sqli) >= 10 and len(false_positives) >= 10:
                break

    print("\n" + "="*50)
    print("TOP 10 MẪU SQLi BỊ BỎ SÓT (MISSING ATTACKS)")
    print("="*50)
    if not missed_sqli: print("Không tìm thấy mẫu nào (hoặc model chưa học được gì).")
    for p in missed_sqli:
        print(f"-> {p}")

    print("\n" + "="*50)
    print("TOP 10 MẪU BENIGN BỊ NHẦM LÀ SQLi (FALSE ALARMS)")
    print("="*50)
    if not false_positives: print("Không tìm thấy mẫu nào.")
    for p in false_positives:
        print(f"-> {p}")

if __name__ == "__main__":
    debug_analysis()
