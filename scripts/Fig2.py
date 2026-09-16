import matplotlib.pyplot as plt
import networkx as nx
import os

# Đảm bảo thư mục lưu ảnh tồn tại
output_dir = '../logs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def draw_figure_2_svg():
    # Khởi tạo 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    plt.subplots_adjust(wspace=0.4)

    # ===================== A. NORMAL =====================
    G1 = nx.DiGraph()
    n1_nodes = ["GET", "/index", "HTTP/1.1"]

    G1.add_nodes_from(range(len(n1_nodes)))
    for i in range(len(n1_nodes) - 1):
        G1.add_edge(i, i + 1)

    pos1 = {i: (i, 0) for i in range(len(n1_nodes))}

    axes[0].set_axis_off()
    nx.draw(G1, pos1, ax=axes[0],
            with_labels=True,
            labels={i: n1_nodes[i] for i in range(len(n1_nodes))},
            node_color='#ebf5fb',
            edgecolors='black',
            node_size=2500,
            font_size=10,
            arrowsize=20)

    axes[0].set_title("A. Normal Request\n(Sequential Structure Only)", fontsize=13, fontweight='bold')

    # ===================== B. SQLi =====================
    G2 = nx.DiGraph()
    n2_nodes = ["id=1", "UNION", "SELECT", "password", "FROM", "users"]

    G2.add_nodes_from(range(len(n2_nodes)))
    for i in range(len(n2_nodes) - 1):
        G2.add_edge(i, i + 1)

    # Dependency edge
    G2.add_edge(2, 4)

    pos2 = {i: (i, 0.1 if i % 2 == 0 else -0.1) for i in range(len(n2_nodes))}
    node_colors_2 = ['#f8c471' if i in [2, 3, 4] else '#d1f2eb' for i in range(len(n2_nodes))]

    axes[1].set_axis_off()
    nx.draw_networkx_nodes(G2, pos2, ax=axes[1], node_color=node_colors_2, edgecolors='black', node_size=2500)
    nx.draw_networkx_labels(G2, pos2, ax=axes[1], labels={i: n2_nodes[i] for i in range(len(n2_nodes))}, font_size=10)

    # Sequential edges
    nx.draw_networkx_edges(G2, pos2, ax=axes[1],
                           edgelist=[(i, i + 1) for i in range(len(n2_nodes) - 1)],
                           width=1, arrowsize=15)

    # Dependency edge
    nx.draw_networkx_edges(G2, pos2, ax=axes[1],
                           edgelist=[(2, 4)],
                           width=3,
                           edge_color='green',
                           connectionstyle="arc3,rad=0.6")

    axes[1].set_title("B. SQL Injection\n(Logical Dependency Edges)", fontsize=13, fontweight='bold')

    # ===================== C. XSS =====================
    G3 = nx.DiGraph()
    n3_nodes = ["<scr", "/**/", "ipt>", "alert", "(", "1", ")"]

    G3.add_nodes_from(range(len(n3_nodes)))
    for i in range(len(n3_nodes) - 1):
        G3.add_edge(i, i + 1)

    # Skip edge
    G3.add_edge(0, 2)

    pos3 = {i: (i, 0) for i in range(len(n3_nodes))}
    node_colors_3 = ['#fadbd8' if i in [0, 2, 3] else '#fdedec' for i in range(len(n3_nodes))]

    axes[2].set_axis_off()
    nx.draw_networkx_nodes(G3, pos3, ax=axes[2], node_color=node_colors_3, edgecolors='black', node_size=2500)
    nx.draw_networkx_labels(G3, pos3, ax=axes[2], labels={i: n3_nodes[i] for i in range(len(n3_nodes))}, font_size=10)

    # Sequential edges
    nx.draw_networkx_edges(G3, pos3, ax=axes[2],
                           edgelist=[(i, i + 1) for i in range(len(n3_nodes) - 1)],
                           width=1, arrowsize=15)

    # Skip edge
    nx.draw_networkx_edges(G3, pos3, ax=axes[2],
                           edgelist=[(0, 2)],
                           width=2,
                           edge_color='red',
                           style='dashed',
                           connectionstyle="arc3,rad=-0.5")

    axes[2].set_title("C. XSS with Obfuscation\n(Skip Edges k=2)", fontsize=13, fontweight='bold')

    # Save SVG
    save_path = os.path.join(output_dir, 'Figure_2_Topology_Comparison.svg')
    plt.tight_layout()
    plt.savefig(save_path, format='svg', dpi=300)

    print(f"Saved Figure 2 at: {save_path}")
    plt.show()

draw_figure_2_svg()