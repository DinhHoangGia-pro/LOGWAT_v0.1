import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_figure_3():

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    # ===== Khối 1 =====
    ax.add_patch(
        patches.Rectangle(
            (5, 35),
            32,
            18,
            facecolor="#ebf5fb",
            edgecolor="black",
            linewidth=1.5,
        )
    )

    ax.text(
        21,
        47,
        "Benign Request Frames\n(CSIC 2010 Dataset)",
        ha="center",
        fontweight="bold",
        fontsize=10,
    )

    ax.text(
        21,
        39,
        "Headers: User-Agent, Cookie\nContext: Clean Protocol Structure",
        ha="center",
        fontsize=9,
    )

    # ===== Khối 2 =====
    ax.add_patch(
        patches.Rectangle(
            (5, 7),
            32,
            18,
            facecolor="#fdedec",
            edgecolor="black",
            linewidth=1.5,
        )
    )

    ax.text(
        21,
        19,
        "Malicious Payloads\n(XSS_dataset.csv)",
        ha="center",
        fontweight="bold",
        fontsize=10,
    )

    ax.text(
        21,
        11,
        "14,000 script patterns\n(onerror, alert, eval...)",
        ha="center",
        fontsize=9,
    )

    # ===== Khối 3 =====
    ax.add_patch(
        patches.FancyBboxPatch(
            (52, 21),
            16,
            18,
            boxstyle="round,pad=0.3",
            facecolor="#f7dc6f",
            edgecolor="black",
            linewidth=1.5,
        )
    )

    ax.text(
        60,
        33,
        "Behavioral Injection\nAugmentation (BIA)",
        ha="center",
        fontweight="bold",
        fontsize=10,
    )

    ax.text(
        60,
        25,
        "Context-Preserving Poisoning",
        ha="center",
        fontsize=9,
        style="italic",
    )

    # ===== Khối 4 =====
    ax.add_patch(
        patches.Rectangle(
            (82, 21),
            16,
            18,
            facecolor="#d4efdf",
            edgecolor="black",
            linewidth=1.5,
        )
    )

    ax.text(
        90,
        31,
        "Balanced\nAttack Graphs\n(42,000 total)",
        ha="center",
        fontweight="bold",
        fontsize=10,
    )

    ax.text(
        90,
        23,
        "14k Benign | 14k SQLi | 14k XSS",
        ha="center",
        fontsize=8,
    )

    # ===== Arrow =====
    ax.annotate(
        "",
        xy=(52, 30),
        xytext=(37, 44),
        arrowprops=dict(arrowstyle="->", lw=2),
    )

    ax.annotate(
        "",
        xy=(52, 30),
        xytext=(37, 16),
        arrowprops=dict(arrowstyle="->", lw=2),
    )

    ax.annotate(
        "",
        xy=(82, 30),
        xytext=(68, 30),
        arrowprops=dict(arrowstyle="->", lw=2),
    )

    plt.savefig("../logs/Figure_3_BIA_Workflow.png", dpi=300)
    plt.close()


draw_figure_3()
