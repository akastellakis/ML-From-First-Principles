"""Generate conceptual illustrations for the Decision Trees notebook."""

import os
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations"), exist_ok=True)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── Shared style ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.grid": False,
    "figure.dpi": 150,
})

COLORS = {
    "blue": "#4A90D9",
    "coral": "#E8755A",
    "green": "#5DAE8B",
    "orange": "#E8A838",
    "purple": "#9B72CF",
    "gray": "#888888",
    "lightblue": "#D0E4F7",
    "lightcoral": "#FADBD8",
    "lightgreen": "#D5F0E5",
    "lightorange": "#FDF0D5",
}


# ═══════════════════════════════════════════════════════════════════
# Fig 1: Entropy, Gini, and Information Content
# ═══════════════════════════════════════════════════════════════════
def fig1_entropy_information():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    p = np.linspace(0.001, 0.999, 500)

    # Left panel: Binary entropy curve with annotations
    ax = axes[0]
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    ax.plot(p, h, linewidth=2.5, color=COLORS["blue"])
    ax.fill_between(p, h, alpha=0.08, color=COLORS["blue"])

    # Annotate key points
    ax.annotate(
        "Maximum uncertainty\n1 bit needed",
        xy=(0.5, 1.0), xytext=(0.72, 0.85),
        fontsize=10, ha="center",
        arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["gray"], alpha=0.9),
    )
    ax.annotate(
        "Pure node\n0 bits needed",
        xy=(0.02, 0.0), xytext=(0.18, 0.25),
        fontsize=10, ha="center",
        arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["gray"], alpha=0.9),
    )
    ax.annotate(
        "Pure node\n0 bits needed",
        xy=(0.98, 0.0), xytext=(0.82, 0.25),
        fontsize=10, ha="center",
        arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["gray"], alpha=0.9),
    )

    ax.set_xlabel(r"$P(\mathrm{class} = 1)$", fontsize=12)
    ax.set_ylabel(r"$H(Y)$ (bits)", fontsize=12)
    ax.set_title("Binary Entropy", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(0, color="black", linewidth=0.5)

    # Right panel: Entropy vs Gini comparison
    ax = axes[1]
    g = 2 * p * (1 - p)
    misclass = np.minimum(p, 1 - p)

    ax.plot(p, h, linewidth=2.5, color=COLORS["blue"], label=r"Entropy $H(p)$")
    ax.plot(p, g, linewidth=2.5, color=COLORS["coral"], label=r"Gini $2p(1-p)$")
    ax.plot(p, misclass, linewidth=2, color=COLORS["green"], linestyle="--",
            label="Misclassification error")

    ax.set_xlabel(r"$P(\mathrm{class} = 1)$", fontsize=12)
    ax.set_ylabel("Impurity", fontsize=12)
    ax.set_title("Impurity Measures Compared", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=10, loc="upper center", framealpha=0.9)
    ax.axhline(0, color="black", linewidth=0.5)

    # Add text box explaining the relationship
    ax.text(
        0.5, -0.18,
        r"Gini $\approx$ 2nd-order Taylor expansion of Entropy around $p = 1$",
        transform=ax.transAxes, fontsize=9.5, ha="center", style="italic",
        color=COLORS["gray"],
    )

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations", "fig1_entropy_information.png"), bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved fig1_entropy_information.png")


# ═══════════════════════════════════════════════════════════════════
# Fig 2: Recursive Partitioning (feature space + tree structure)
# ═══════════════════════════════════════════════════════════════════
def fig2_recursive_partitioning():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left panel: 2D feature space with rectangular splits ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Draw split lines (simulated CART splits)
    # Split 1: x1 <= 5 (vertical)
    ax.axvline(5, color="black", linewidth=2, linestyle="-")
    ax.text(5.0, 9.5, r"$x_1 \leq 5$", fontsize=11, ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85))

    # Split 2: left side, x2 <= 6 (horizontal)
    ax.plot([0, 5], [6, 6], color=COLORS["blue"], linewidth=2, linestyle="-")
    ax.text(2.5, 6.3, r"$x_2 \leq 6$", fontsize=10, ha="center", color=COLORS["blue"],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS["blue"], alpha=0.85))

    # Split 3: right side, x2 <= 3 (horizontal)
    ax.plot([5, 10], [3, 3], color=COLORS["coral"], linewidth=2, linestyle="-")
    ax.text(7.5, 3.3, r"$x_2 \leq 3$", fontsize=10, ha="center", color=COLORS["coral"],
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS["coral"], alpha=0.85))

    # Color regions
    # Region A: x1<=5, x2<=6 (bottom-left)
    ax.fill_between([0, 5], 0, 6, alpha=0.15, color=COLORS["blue"])
    ax.text(2.5, 3, "A", fontsize=18, ha="center", va="center",
            fontweight="bold", color=COLORS["blue"])

    # Region B: x1<=5, x2>6 (top-left)
    ax.fill_between([0, 5], 6, 10, alpha=0.15, color=COLORS["coral"])
    ax.text(2.5, 8, "B", fontsize=18, ha="center", va="center",
            fontweight="bold", color=COLORS["coral"])

    # Region C: x1>5, x2<=3 (bottom-right)
    ax.fill_between([5, 10], 0, 3, alpha=0.15, color=COLORS["green"])
    ax.text(7.5, 1.5, "C", fontsize=18, ha="center", va="center",
            fontweight="bold", color=COLORS["green"])

    # Region D: x1>5, x2>3 (top-right)
    ax.fill_between([5, 10], 3, 10, alpha=0.15, color=COLORS["orange"])
    ax.text(7.5, 6.5, "D", fontsize=18, ha="center", va="center",
            fontweight="bold", color=COLORS["orange"])

    ax.set_xlabel(r"Feature $x_1$", fontsize=12)
    ax.set_ylabel(r"Feature $x_2$", fontsize=12)
    ax.set_title("Feature Space Partitioning", fontsize=13, fontweight="bold")

    # ---- Right panel: Corresponding tree structure ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Corresponding Decision Tree", fontsize=13, fontweight="bold")

    # Node positions
    nodes = {
        "root": (5, 9),
        "left": (2.5, 6),
        "right": (7.5, 6),
        "ll": (1.2, 3),
        "lr": (3.8, 3),
        "rl": (6.2, 3),
        "rr": (8.8, 3),
    }

    # Draw edges
    for parent, child in [("root", "left"), ("root", "right"),
                          ("left", "ll"), ("left", "lr"),
                          ("right", "rl"), ("right", "rr")]:
        px, py = nodes[parent]
        cx, cy = nodes[child]
        ax.plot([px, cx], [py - 0.5, cy + 0.7], color=COLORS["gray"], linewidth=1.5,
                zorder=1)

    # Edge labels
    ax.text(3.3, 7.8, "Yes", fontsize=9, color=COLORS["gray"], ha="center")
    ax.text(6.7, 7.8, "No", fontsize=9, color=COLORS["gray"], ha="center")
    ax.text(1.5, 4.7, "Yes", fontsize=9, color=COLORS["gray"], ha="center")
    ax.text(3.5, 4.7, "No", fontsize=9, color=COLORS["gray"], ha="center")
    ax.text(6.5, 4.7, "Yes", fontsize=9, color=COLORS["gray"], ha="center")
    ax.text(8.5, 4.7, "No", fontsize=9, color=COLORS["gray"], ha="center")

    # Draw internal nodes (rounded boxes)
    for name, (x, y), label in [
        ("root", nodes["root"], r"$x_1 \leq 5$?"),
        ("left", nodes["left"], r"$x_2 \leq 6$?"),
        ("right", nodes["right"], r"$x_2 \leq 3$?"),
    ]:
        bbox = FancyBboxPatch(
            (x - 1.2, y - 0.5), 2.4, 1.0,
            boxstyle="round,pad=0.15", facecolor="white",
            edgecolor="black", linewidth=1.5, zorder=2,
        )
        ax.add_patch(bbox)
        ax.text(x, y, label, fontsize=11, ha="center", va="center", zorder=3,
                fontweight="bold")

    # Draw leaf nodes (colored boxes)
    for (x, y), label, color in [
        (nodes["ll"], "A", COLORS["blue"]),
        (nodes["lr"], "B", COLORS["coral"]),
        (nodes["rl"], "C", COLORS["green"]),
        (nodes["rr"], "D", COLORS["orange"]),
    ]:
        bbox = FancyBboxPatch(
            (x - 0.7, y - 0.5), 1.4, 1.0,
            boxstyle="round,pad=0.15", facecolor=color,
            edgecolor="black", linewidth=1.5, alpha=0.3, zorder=2,
        )
        ax.add_patch(bbox)
        ax.text(x, y, label, fontsize=14, ha="center", va="center", zorder=3,
                fontweight="bold", color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations", "fig2_recursive_partitioning.png"), bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved fig2_recursive_partitioning.png")


# ═══════════════════════════════════════════════════════════════════
# Fig 3: Pruning and Bias-Variance Tradeoff
# ═══════════════════════════════════════════════════════════════════
def fig3_pruning_bias_variance():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- Left panel: Full tree vs pruned tree (schematic) ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Full Tree vs Pruned Tree", fontsize=13, fontweight="bold")

    # Full tree (left side)
    ax.text(2.5, 9.7, "Full (overfit)", fontsize=11, ha="center", fontweight="bold",
            color=COLORS["coral"])

    # Draw a bushy tree schematically
    def draw_mini_tree(ax, cx, cy, depth, spread, max_depth, color, alpha=1.0):
        if depth >= max_depth:
            ax.plot(cx, cy, "s", markersize=5, color=color, alpha=alpha, zorder=3)
            return
        ax.plot(cx, cy, "o", markersize=6, color=color, alpha=alpha, zorder=3)
        dx = spread / (2 ** depth)
        dy = 1.2
        for child_x in [cx - dx, cx + dx]:
            ax.plot([cx, child_x], [cy, cy - dy], color=color, linewidth=1, alpha=alpha,
                    zorder=2)
            draw_mini_tree(ax, child_x, cy - dy, depth + 1, spread, max_depth, color, alpha)

    draw_mini_tree(ax, 2.5, 9.0, 0, 3.5, 4, COLORS["coral"], alpha=0.7)

    # Pruned tree (right side)
    ax.text(7.5, 9.7, "Pruned (generalizes)", fontsize=11, ha="center", fontweight="bold",
            color=COLORS["green"])

    draw_mini_tree(ax, 7.5, 9.0, 0, 3.5, 2, COLORS["green"], alpha=0.8)

    # Arrow between them
    ax.annotate(
        "", xy=(5.8, 6.5), xytext=(4.2, 6.5),
        arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=2),
    )
    ax.text(5.0, 6.9, "Prune", fontsize=10, ha="center", color=COLORS["gray"],
            fontweight="bold")

    # Labels below
    ax.text(2.5, 0.8, "Depth ~20, ~200 leaves\nMemorizes training data",
            fontsize=9, ha="center", color=COLORS["gray"], style="italic")
    ax.text(7.5, 0.8, "Depth ~5, ~30 leaves\nCaptures true patterns",
            fontsize=9, ha="center", color=COLORS["gray"], style="italic")

    # ---- Right panel: Bias-variance vs depth curve ----
    ax = axes[1]

    depths = np.linspace(1, 25, 200)

    # Simulated curves
    train_err = 0.15 * np.exp(-0.3 * depths) + 0.001
    test_err = 0.15 * np.exp(-0.3 * depths) + 0.005 + 0.0008 * (depths - 5) ** 2 * (depths > 5)
    test_err = np.where(depths <= 5, 0.15 * np.exp(-0.3 * depths) + 0.01, test_err)

    ax.plot(depths, train_err, linewidth=2.5, color=COLORS["blue"], label="Training error")
    ax.plot(depths, test_err, linewidth=2.5, color=COLORS["coral"], label="Test error")

    # Optimal depth
    opt_idx = np.argmin(test_err)
    opt_depth = depths[opt_idx]
    ax.axvline(opt_depth, color=COLORS["green"], linewidth=1.5, linestyle="--", alpha=0.7)
    ax.annotate(
        f"Optimal depth",
        xy=(opt_depth, test_err[opt_idx]), xytext=(opt_depth + 5, test_err[opt_idx] + 0.03),
        fontsize=10, ha="center",
        arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["green"], alpha=0.9),
    )

    # Shade regions
    ax.axvspan(1, 4, alpha=0.07, color=COLORS["blue"])
    ax.axvspan(4, 8, alpha=0.07, color=COLORS["green"])
    ax.axvspan(8, 25, alpha=0.07, color=COLORS["coral"])

    ax.text(2.5, 0.17, "Underfitting", fontsize=9, ha="center", color=COLORS["blue"],
            fontweight="bold")
    ax.text(6, 0.17, "Sweet spot", fontsize=9, ha="center", color=COLORS["green"],
            fontweight="bold")
    ax.text(16, 0.17, "Overfitting", fontsize=9, ha="center", color=COLORS["coral"],
            fontweight="bold")

    ax.set_xlabel("Tree Depth", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.set_title("Bias-Variance Tradeoff", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_xlim(1, 25)
    ax.set_ylim(0, 0.20)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations", "fig3_pruning_bias_variance.png"), bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved fig3_pruning_bias_variance.png")


# ═══════════════════════════════════════════════════════════════════
# Generate all figures
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig1_entropy_information()
    fig2_recursive_partitioning()
    fig3_pruning_bias_variance()
    print("All illustrations generated.")
