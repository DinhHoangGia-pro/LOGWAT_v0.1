import os, pickle, torch, shutil
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm, global_max_pool, JumpingKnowledge
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

HIN_DIR = "/home/dxthanh/hin_web_vulne"
DATA_PATH = os.path.join(HIN_DIR, "data/web_graphs.pkl")
MODEL_SAVE_PATH = os.path.join(HIN_DIR, "data/models_pretrained/best_web_gnn.pth")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super().__init__()
        self.alpha, self.gamma, self.weight = alpha, gamma, weight
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt)**self.gamma * ce_loss).mean()

class HeavyWebGNN(nn.Module):
    def __init__(self, in_channels=64, hidden_dim=256, out_channels=3):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_dim // 4, heads=4)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4)
        self.bn2 = BatchNorm(hidden_dim)
        self.jk = JumpingKnowledge(mode='cat')
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.5), # Tăng dropout để chống học vẹt
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, x, edge_index, batch):
        x1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        x2 = F.elu(self.bn2(self.conv2(x1, edge_index)))
        x_jk = self.jk([x1, x2])
        return self.classifier(global_max_pool(x_jk, batch))

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(DATA_PATH, 'rb') as f: data_pkl = pickle.load(f)
    
    train_data, test_data = train_test_split(data_pkl['graphs'], test_size=0.15, random_state=42)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128)

    model = HeavyWebGNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    criterion = FocalLoss(gamma=2)
    best_acc = 0

    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data.x, data.edge_index, data.batch), data.y)
            loss.backward(); optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
                correct += (pred == data.y).sum().item()
        
        acc = correct / len(test_data)
        scheduler.step(acc)
        
        print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   [V] New Best Saved: {best_acc:.4f}")

if __name__ == "__main__": train()
import os, pickle, torch, shutil
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm, global_max_pool, JumpingKnowledge
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

HIN_DIR = "/home/dxthanh/hin_web_vulne"
DATA_PATH = os.path.join(HIN_DIR, "data/web_graphs.pkl")
MODEL_SAVE_PATH = os.path.join(HIN_DIR, "data/models_pretrained/best_web_gnn.pth")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super().__init__()
        self.alpha, self.gamma, self.weight = alpha, gamma, weight
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt)**self.gamma * ce_loss).mean()

class HeavyWebGNN(nn.Module):
    def __init__(self, in_channels=64, hidden_dim=256, out_channels=3):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_dim // 4, heads=4)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 4, heads=4)
        self.bn2 = BatchNorm(hidden_dim)
        self.jk = JumpingKnowledge(mode='cat')
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.5), # Tăng dropout để chống học vẹt
            nn.Linear(hidden_dim, out_channels)
        )

    def forward(self, x, edge_index, batch):
        x1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        x2 = F.elu(self.bn2(self.conv2(x1, edge_index)))
        x_jk = self.jk([x1, x2])
        return self.classifier(global_max_pool(x_jk, batch))

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(DATA_PATH, 'rb') as f: data_pkl = pickle.load(f)
    
    train_data, test_data = train_test_split(data_pkl['graphs'], test_size=0.15, random_state=42)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=128)

    model = HeavyWebGNN().to(device)
    # LR thấp hơn và Weight Decay mạnh hơn để ổn định Acc
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    # Scheduler: Nếu Acc không tăng sau 5 epoch, giảm LR đi một nửa
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    criterion = FocalLoss(gamma=2)
    best_acc = 0

    for epoch in range(1, 101): # Chạy 100 epoch để scheduler có đất diễn
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data.x, data.edge_index, data.batch), data.y)
            loss.backward(); optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                pred = model(data.x, data.edge_index, data.batch).argmax(dim=1)
                correct += (pred == data.y).sum().item()
        
        acc = correct / len(test_data)
        scheduler.step(acc) # Cập nhật LR dựa trên Acc
        
        print(f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   [V] New Best Saved: {best_acc:.4f}")

if __name__ == "__main__": train()
