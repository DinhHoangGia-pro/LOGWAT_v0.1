import torch
import pickle
import os
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIG ---
HIN_DIR = "/home/dxthanh/hin_web_vulne"
DATA_PATH = os.path.join(HIN_DIR, "data/web_graphs.pkl")
MODEL_PATH = os.path.join(HIN_DIR, "data/models_pretrained/best_web_gnn.pth")

def check_data_overlap():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(DATA_PATH, 'rb') as f:
        data_pkl = pickle.load(f)
    
    dataset = data_pkl['graphs']
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Khởi tạo model (Dùng kiến trúc ResGNN bạn vừa chạy gần nhất)
    from train_web_gnn import WebAttackResGNN
    model = WebAttackResGNN(data_pkl['vocab_size']).to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    y_true = []
    y_pred = []

    print("[*] Đang phân tích lỗi trên toàn bộ tập dữ liệu...")
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            y_pred.extend(out.argmax(dim=1).cpu().numpy())
            y_true.extend(data.y.cpu().numpy())

    print("\n=== CHI TIẾT PHÂN LOẠI ===")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'SQLi', 'XSS']))
    
    print("\n=== MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ===")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Giải thích kết quả
    overlap_sqli_xss = cm[1][2] + cm[2][1]
    print(f"\n[!] Số mẫu SQLi và XSS bị nhận nhầm lẫn nhau: {overlap_sqli_xss}")
    if overlap_sqli_xss > 1000:
        print("=> KẾT LUẬN: Dữ liệu bị Overlap nặng. GNN không phân biệt được cấu trúc đồ thị của SQLi và XSS.")

if __name__ == "__main__":
    check_data_overlap()
