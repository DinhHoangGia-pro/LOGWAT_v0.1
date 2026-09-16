import matplotlib.pyplot as plt
import networkx as nx
import os


def draw_updated_figure_2():
    # Khởi tạo 3 subplots đại diện cho 3 kịch bản
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    plt.subplots_adjust(wspace=0.3)

    # Cấu hình chung cho font
    plt.rcParams['font.family'] = 'sans-serif'

    # ===================== A. NORMAL REQUEST =====================
    # Minh họa cấu trúc tuần tự đơn giản của request sạch
    G1 = nx.DiGraph()
    n1_labels = ["GET", "/index", "HTTP/1.1"]
    G1.add_nodes_from(range(len(n1_labels)))
    edges_seq1 = [(i, i + 1) for i in range(len(n1_labels) - 1)]
    
    pos1 = {i: (i, 0) for i in range(len(n1_labels))}
    axes[0].set_axis_off()
    axes[0].set_title("A. Normal Request (Sequential)", fontsize=14, fontweight='bold', pad=20)
    
    nx.draw_networkx_nodes(G1, pos1, ax=axes[0], node_color='#ebf5fb', edgecolors='black', node_size=3000)
    nx.draw_networkx_labels(G1, pos1, ax=axes[0], labels={i: n1_labels[i] for i in range(len(n1_labels))}, font_size=11)
    nx.draw_networkx_edges(G1, pos1, ax=axes[0], edgelist=edges_seq1, width=1.5, arrowsize=20)

    # ===================== B. SQL INJECTION (Semantic Focus) =====================
    G2 = nx.DiGraph()
    n2_labels = ["1'", "UNION", "SELECT", "junk_1", "junk_2", "FROM", "users"]
    G2.add_nodes_from(range(len(n2_labels)))
    
    edges_seq2 = [(i, i + 1) for i in range(len(n2_labels) - 1)]
    edges_semantic2 = [(2, 5)] 

    pos2 = {i: (i, 0) for i in range(len(n2_labels))}
    axes[1].set_axis_off()
    axes[1].set_title("B. SQL Injection (Semantic Dependency)", fontsize=14, fontweight='bold', pad=20)
    
    nx.draw_networkx_nodes(G2, pos2, ax=axes[1], node_color='#e8f8f5', edgecolors='black', node_size=3000)
    nx.draw_networkx_labels(G2, pos2, ax=axes[1], labels={i: n2_labels[i] for i in range(len(n2_labels))}, font_size=11)
    nx.draw_networkx_edges(G2, pos2, ax=axes[1], edgelist=edges_seq2, width=1, edge_color='gray', arrowsize=15)
    nx.draw_networkx_edges(G2, pos2, ax=axes[1], edgelist=edges_semantic2, 
                           width=2.5, edge_color='red', style='--', 
                           connectionstyle="arc3,rad=0.5", arrowsize=25, label="Semantic Edge")

    # ===================== C. OBFUSCATED XSS (Skip + Semantic) =====================
    G3 = nx.DiGraph()
    n3_labels = ["<scr", "/**/", "ipt>", "alert", "(", "1", ")"]
    G3.add_nodes_from(range(len(n3_labels)))
    
    edges_seq3 = [(i, i + 1) for i in range(len(n3_labels) - 1)]
    edges_special3 = [(0, 2), (2, 3)]

    pos3 = {i: (i, 0) for i in range(len(n3_labels))}
    axes[2].set_axis_off()
    axes[2].set_title("C. Obfuscated XSS (Skip & Semantic)", fontsize=14, fontweight='bold', pad=20)
    
    nx.draw_networkx_nodes(G3, pos3, ax=axes[2], node_color='#fef9e7', edgecolors='black', node_size=3000)
    nx.draw_networkx_labels(G3, pos3, ax=axes[2], labels={i: n3_labels[i] for i in range(len(n3_labels))}, font_size=11)
    nx.draw_networkx_edges(G3, pos3, ax=axes[2], edgelist=edges_seq3, width=1, edge_color='gray', arrowsize=15)
    nx.draw_networkx_edges(G3, pos3, ax=axes[2], edgelist=edges_special3, 
                           width=2.5, edge_color='blue', style='--', 
                           connectionstyle="arc3,rad=-0.4", arrowsize=25)

    # Lưu ảnh chất lượng cao
    output_path = os.path.join(output_dir, "Figure_2_Semantic_HIN.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[V] Đã tạo xong hình 2 tại: {output_path}")
    plt.show()

if __name__ == "__main__":
    draw_updated_figure_2()
