import matplotlib.pyplot as plt
import networkx as nx
import os


def draw_figure_2():

    os.makedirs("../logs", exist_ok=True)

    # Khởi tạo 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    plt.subplots_adjust(wspace=0.4)

    # =========================
    # A. NORMAL REQUEST
    # =========================
    G1 = nx.DiGraph()

    n1_nodes = ["GET", "/login", "id", "=", "1"]

    G1.add_nodes_from(range(len(n1_nodes)))

    for i in range(len(n1_nodes) - 1):
        G1.add_edge(i, i + 1)

    pos1 = {i: (i, 0) for i in range(len(n1_nodes))}

    nx.draw(
        G1,
        pos1,
        ax=axes[0],
        with_labels=True,
        labels={i: n1_nodes[i] for i in range(len(n1_nodes))},
        node_color="#ebf5fb",
        edgecolors="black",
        node_size=2500,
        font_size=10,
        arrowsize=20,
    )

    axes[0].set_title(
        "A. Normal Request\n(Sequential Structure Only)",
        fontsize=13,
        fontweight="bold",
    )

    # =========================
    # B. SQL Injection
    # =========================
    G2 = nx.DiGraph()

    n2_nodes = ["GET", "/login", "id", "=", "1", "UNION", "SELECT", "*", "FROM", "users"]

    G2.add_nodes_from(range(len(n2_nodes)))

    # sequential edges
    for i in range(len(n2_nodes) - 1):
        G2.add_edge(i, i + 1)

    # dependency edge SELECT -> FROM
    G2.add_edge(6, 8)

    pos2 = {i: (i, 0.1 if i % 2 == 0 else -0.1) for i in range(len(n2_nodes))}

    node_colors_2 = [
        "#f8c471" if i in [5, 6, 8] else "#d1f2eb"
        for i in range(len(n2_nodes))
    ]

    nx.draw_networkx_nodes(
        G2,
        pos2,
        ax=axes[1],
        node_color=node_colors_2,
        edgecolors="black",
        node_size=2500,
    )

    nx.draw_networkx_labels(
        G2,
        pos2,
        ax=axes[1],
        labels={i: n2_nodes[i] for i in range(len(n2_nodes))},
        font_size=10,
    )

    nx.draw_networkx_edges(
        G2,
        pos2,
        ax=axes[1],
        edgelist=[(i, i + 1) for i in range(len(n2_nodes) - 1)],
        width=1,
        arrowsize=15,
    )

    nx.draw_networkx_edges(
        G2,
        pos2,
        ax=axes[1],
        edgelist=[(6, 8)],
        width=3,
        edge_color="green",
        connectionstyle="arc3,rad=0.5",
    )

    axes[1].set_title(
        "B. SQL Injection\n(Logical Dependency Edges)",
        fontsize=13,
        fontweight="bold",
    )

    # =========================
    # C. XSS with Skip Edge
    # =========================
    G3 = nx.DiGraph()

    n3_nodes = ["<scr", "/**/", "ipt>", "alert", "(", "1", ")"]

    G3.add_nodes_from(range(len(n3_nodes)))

    for i in range(len(n3_nodes) - 1):
        G3.add_edge(i, i + 1)

    # skip edge k=2
    G3.add_edge(0, 2)

    pos3 = {i: (i, 0) for i in range(len(n3_nodes))}

    node_colors_3 = [
        "#fadbd8" if i in [0, 2, 3] else "#fdedec"
        for i in range(len(n3_nodes))
    ]

    nx.draw_networkx_nodes(
        G3,
        pos3,
        ax=axes[2],
        node_color=node_colors_3,
        edgecolors="black",
        node_size=2500,
    )

    nx.draw_networkx_labels(
        G3,
        pos3,
        ax=axes[2],
        labels={i: n3_nodes[i] for i in range(len(n3_nodes))},
        font_size=10,
    )

    nx.draw_networkx_edges(
        G3,
        pos3,
        ax=axes[2],
        edgelist=[(i, i + 1) for i in range(len(n3_nodes) - 1)],
        width=1,
        arrowsize=15,
    )

    nx.draw_networkx_edges(
        G3,
        pos3,
        ax=axes[2],
        edgelist=[(0, 2)],
        width=2,
        edge_color="red",
        style="dashed",
        connectionstyle="arc3,rad=-0.4",
    )

    axes[2].set_title(
        "C. XSS with Obfuscation\n(Skip Edges k=2)",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig("../logs/Figure_2_Topology_Comparison.png", dpi=300)
    plt.show()


draw_figure_2()
