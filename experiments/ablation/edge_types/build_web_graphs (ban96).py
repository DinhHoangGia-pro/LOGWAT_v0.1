import pandas as pd
import torch
import pickle
import os
import re
import math
from urllib.parse import unquote
from torch_geometric.data import Data
from tqdm import tqdm

HIN_DIR = "/home/dxthanh/hin_web_vulne"
INPUT_CSV = os.path.join(HIN_DIR, "data/augmented_web_attack.csv")
OUTPUT_GRAPH_FILE = os.path.join(HIN_DIR, "data/web_graphs.pkl")

def get_entropy(text):
    if not text or len(text) == 0: return 0
    probs = [float(text.count(c)) / len(text) for c in set(text)]
    return -sum([p * math.log(p, 2) for p in probs])

def web_security_tokenizer(text):
    text = unquote(str(text)).lower()
    text = re.sub(r"(['\"#;%\(\)\-\+\/\*<>=\[\]\{\},\.@])", r" \1 ", text)
    return re.findall(r"[\w']+|[^\w\s]", text)

def build_web_graphs():
    print("--- [1/2] BUILD GRAPH: 64-FEATURES (STABLE VERSION) ---")
    df = pd.read_csv(INPUT_CSV)
    
    danger_chars = ["'", '"', ";", "--", "#", "/*", "(", ")", "<", ">", "=", "+", "%", ".", "@"]
    sql_kw = {'select', 'union', 'where', 'from', 'insert', 'drop', 'limit', 'exec', 'null'}
    web_kw = {'script', 'alert', 'onerror', 'eval', 'src', 'href', 'javascript'}

    processed_graphs = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        tokens = web_security_tokenizer(row['content'])
        num_nodes = len(tokens)
        
        x_list = []
        for i, t in enumerate(tokens):
            f_stat = [len(t)/20.0, i/(num_nodes+1), get_entropy(t)/8.0, 1.0 if t.isdigit() else 0.0,
                      sum(c.isdigit() for c in t)/(len(t)+1), sum(not c.isalnum() for c in t)/(len(t)+1),
                      1.0 if len(t) > 12 else 0.0, 1.0 if re.search(r"[0-9a-f]{4,}", t) else 0.0]
            f_char = [1.0 if c in t else 0.0 for c in danger_chars]
            f_sql = [1.0 if t == kw else 0.0 for kw in sql_kw]
            f_web = [1.0 if t == kw else 0.0 for kw in web_kw]
            
            feat = f_stat + f_char + f_sql + f_web
            feat += [0.0] * (64 - len(feat)) 
            x_list.append(feat)
        
        x = torch.tensor(x_list, dtype=torch.float)
        edges = []
        for i in range(num_nodes):
            if i < num_nodes - 1: edges.extend([[i, i+1], [i+1, i]])
            if i < num_nodes - 2: edges.extend([[i, i+2], [i+2, i]]) 
        
        edge_index = torch.tensor(edges if edges else [[0,0]], dtype=torch.long).t().contiguous()
        processed_graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([row['attack_type']], dtype=torch.long)))

    with open(OUTPUT_GRAPH_FILE, 'wb') as f:
        pickle.dump({'graphs': processed_graphs}, f)
    print(f"\n[V] Đồ thị sẵn sàng. Nodes: {len(processed_graphs)}")

if __name__ == "__main__": build_web_graphs()
import pandas as pd
import torch
import pickle
import os
import re
import math
from urllib.parse import unquote
from torch_geometric.data import Data
from tqdm import tqdm

HIN_DIR = "/home/dxthanh/hin_web_vulne"
INPUT_CSV = os.path.join(HIN_DIR, "data/augmented_web_attack.csv")
OUTPUT_GRAPH_FILE = os.path.join(HIN_DIR, "data/web_graphs.pkl")

def get_entropy(text):
    if not text or len(text) == 0: return 0
    probs = [float(text.count(c)) / len(text) for c in set(text)]
    return -sum([p * math.log(p, 2) for p in probs])

def web_security_tokenizer(text):
    text = unquote(str(text)).lower()
    text = re.sub(r"(['\"#;%\(\)\-\+\/\*<>=\[\]\{\},\.@])", r" \1 ", text)
    return re.findall(r"[\w']+|[^\w\s]", text)

def build_web_graphs():
    print("--- [1/2] BUILD GRAPH: 64-FEATURES (STABLE VERSION) ---")
    df = pd.read_csv(INPUT_CSV)
    
    danger_chars = ["'", "\"", ";", "--", "#", "/*", "(", ")", "<", ">", "=", "+", "%", ".", "@"]
    sql_kw = {'select', 'union', 'where', 'from', 'insert', 'drop', 'limit', 'exec', 'null'}
    web_kw = {'script', 'alert', 'onerror', 'eval', 'src', 'href', 'javascript'}

    processed_graphs = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        tokens = web_security_tokenizer(row['content'])
        num_nodes = len(tokens)
        
        x_list = []
        for i, t in enumerate(tokens):
            f_stat = [len(t)/20.0, i/(num_nodes+1), get_entropy(t)/8.0, 1.0 if t.isdigit() else 0.0,
                      sum(c.isdigit() for c in t)/(len(t)+1), sum(not c.isalnum() for c in t)/(len(t)+1),
                      1.0 if len(t) > 12 else 0.0, 1.0 if re.search(r"[0-9a-f]{4,}", t) else 0.0]
            f_char = [1.0 if c in t else 0.0 for c in danger_chars]
            f_sql = [1.0 if t == kw else 0.0 for kw in sql_kw]
            f_web = [1.0 if t == kw else 0.0 for kw in web_kw]
            
            feat = f_stat + f_char + f_sql + f_web
            feat += [0.0] * (64 - len(feat)) 
            x_list.append(feat)
        
        x = torch.tensor(x_list, dtype=torch.float)
        edges = []
        for i in range(num_nodes):
            if i < num_nodes - 1: edges.extend([[i, i+1], [i+1, i]])
            if i < num_nodes - 2: edges.extend([[i, i+2], [i+2, i]]) 
        
        edge_index = torch.tensor(edges if edges else [[0,0]], dtype=torch.long).t().contiguous()
        processed_graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([row['attack_type']], dtype=torch.long)))

    with open(OUTPUT_GRAPH_FILE, 'wb') as f:
        pickle.dump({'graphs': processed_graphs}, f)
    print(f"\n[V] Đồ thị sẵn sàng. Nodes: {len(processed_graphs)}")

if __name__ == "__main__": build_web_graphs()
