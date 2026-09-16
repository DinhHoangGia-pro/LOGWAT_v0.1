import os
import pickle
import time
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from src.models.logwat import HeavyWebGNN
from src.utils.config import load_all

cfgs = load_all()
dataset_cfg = cfgs['dataset']
config_cfg = cfgs['config']

HIN_DIR = dataset_cfg.get('hin_dir')
paths = dataset_cfg.get('paths', {})
DATA_PATH = os.path.join(HIN_DIR, paths.get('graphs_pkl', 'data/web_graphs.pkl'))
MODEL_SAVE_PATH = os.path.join(HIN_DIR, paths.get('model_save', 'data/models_pretrained/best_web_gnn.pth'))
LOG_PATH = os.path.join(HIN_DIR, paths.get('logs_dir', 'logs'), 'training_history.log')


def train(num_epochs=None, batch_size=None, lr=None, target_metric='acc'):
    # fill defaults from config if not provided
    training_cfg = config_cfg.get('training', {})
    if num_epochs is None:
        num_epochs = training_cfg.get('epochs', 100)
    if batch_size is None:
        batch_size = training_cfg.get('batch_size', 64)
    if lr is None:
        lr = training_cfg.get('lr', 5e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(DATA_PATH, 'rb') as f:
        data_pkl = pickle.load(f)

    train_data, test_data = train_test_split(data_pkl['graphs'], test_size=0.15, random_state=42)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = HeavyWebGNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=training_cfg.get('weight_decay', 0.1))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    with open(LOG_PATH, 'a') as log_f:
        log_f.write(f"\n--- NEW SESSION: {time.ctime()} ---\n")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            model.train()
            total_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
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
            epoch_time = time.time() - start_time

            log_str = f"Epoch {epoch:03d} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {epoch_time:.1f}s"
            print(log_str)
            log_f.write(log_str + "\n")

            if acc > best_acc:
                best_acc = acc
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"   [V] Saved Best: {best_acc:.4f}")


if __name__ == '__main__':
    train()
