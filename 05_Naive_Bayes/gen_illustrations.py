"""Generate conceptual illustrations for the Naive Bayes notebook."""

import os
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations"), exist_ok=True)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm

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
}


# ═══════════════════════════════════════════════════════════════════
# Fig 1: Naive Independence Assumption
# ═══════════════════════════════════════════════════════════════════
def fig1_naive_independence():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: Full joint (exponential parameters) ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title("Full Joint Model", fontweight="bold", fontsize=14, pad=12)

    # Draw a big node for the joint
    joint_box = FancyBboxPatch((2.0, 4.0), 6, 2.8, boxstyle="round,pad=0.3",
                                facecolor=COLORS["lightcoral"], edgecolor=COLORS["coral"],
                                linewidth=2)
    ax.add_patch(joint_box)
    ax.text(5, 6.0, r"$P(x_1, x_2, \ldots, x_d \mid c)$", ha="center", va="center",
            fontsize=14, fontweight="bold")
    ax.text(5, 4.8, "Exponential parameters", ha="center", va="center",
            fontsize=11, color=COLORS["coral"], style="italic")

    # Feature nodes below
    n_feat = 5
    feat_x = np.linspace(1.5, 8.5, n_feat)
    for i, fx in enumerate(feat_x):
        circle = plt.Circle((fx, 1.8), 0.55, facecolor=COLORS["lightblue"],
                             edgecolor=COLORS["blue"], linewidth=1.5)
        ax.add_patch(circle)
        ax.text(fx, 1.8, f"$x_{i+1}$", ha="center", va="center", fontsize=12)
        # Arrow from joint box down to feature
        ax.annotate("", xy=(fx, 2.35), xytext=(fx, 4.0),
                    arrowprops=dict(arrowstyle="->", color=COLORS["coral"], lw=1.5))

    # Horizontal dependency lines between adjacent features
    for i in range(n_feat - 1):
        ax.annotate("", xy=(feat_x[i+1] - 0.55, 1.8), xytext=(feat_x[i] + 0.55, 1.8),
                    arrowprops=dict(arrowstyle="<->", color=COLORS["gray"],
                                   alpha=0.4, lw=1.5))

    ax.text(5, 0.5, "All features coupled\n(intractable for large d)",
            ha="center", va="center", fontsize=10, style="italic",
            color=COLORS["gray"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                     edgecolor=COLORS["gray"], alpha=0.5))
    ax.axis("off")

    # --- Right: Factored (naive) model ---
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title("Naive Bayes (Factored)", fontweight="bold", fontsize=14, pad=12)

    # Factorization formula at top
    ax.text(5, 9.2, r"$P(\mathbf{x} \mid c) \;=\; \prod_j \; P(x_j \mid c)$",
            ha="center", va="center", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["lightgreen"],
                     edgecolor=COLORS["green"], linewidth=1.5))

    # Class node
    class_circle = plt.Circle((5, 7.0), 0.7, facecolor=COLORS["lightgreen"],
                               edgecolor=COLORS["green"], linewidth=2)
    ax.add_patch(class_circle)
    ax.text(5, 7.0, "$c$", ha="center", va="center", fontsize=15, fontweight="bold")

    # Feature nodes below, each connected only to class
    for i, fx in enumerate(feat_x):
        circle = plt.Circle((fx, 3.8), 0.55, facecolor=COLORS["lightblue"],
                             edgecolor=COLORS["blue"], linewidth=1.5)
        ax.add_patch(circle)
        ax.text(fx, 3.8, f"$x_{i+1}$", ha="center", va="center", fontsize=12)
        ax.annotate("", xy=(fx, 4.35), xytext=(5, 6.3),
                    arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.5))

    # Individual factor labels below nodes
    for i, fx in enumerate(feat_x):
        ax.text(fx, 2.6, f"$P(x_{i+1} \\mid c)$", ha="center", va="center",
                fontsize=10, color=COLORS["blue"])

    ax.text(5, 1.0, "Only $O(Kd)$ parameters\n(tractable)",
            ha="center", va="center", fontsize=10, style="italic",
            color=COLORS["gray"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                     edgecolor=COLORS["gray"], alpha=0.5))
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations", "fig1_naive_independence.png"), dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("  ✓ fig1_naive_independence.png")


# ═══════════════════════════════════════════════════════════════════
# Fig 2: Three NB Variants (Gaussian, Multinomial, Bernoulli)
# ═══════════════════════════════════════════════════════════════════
def fig2_nb_variants():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Gaussian NB ---
    ax = axes[0]
    x = np.linspace(-3, 7, 300)
    ax.fill_between(x, norm.pdf(x, 1, 0.8), alpha=0.3, color=COLORS["blue"])
    ax.fill_between(x, norm.pdf(x, 4, 1.0), alpha=0.3, color=COLORS["coral"])
    ax.plot(x, norm.pdf(x, 1, 0.8), color=COLORS["blue"], linewidth=2)
    ax.plot(x, norm.pdf(x, 4, 1.0), color=COLORS["coral"], linewidth=2)
    ax.set_title("Gaussian NB", fontweight="bold", fontsize=13)
    ax.set_xlabel("Feature value (continuous)")
    ax.set_ylabel("Density")
    # Annotations placed well above the curves to avoid overlap
    ax.annotate(r"$P(x \mid \mathrm{ham})$", xy=(0.6, 0.45), xytext=(-2.5, 0.52),
                fontsize=11, color=COLORS["blue"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=1.2),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                         edgecolor=COLORS["blue"], alpha=0.9))
    ax.annotate(r"$P(x \mid \mathrm{spam})$", xy=(4.4, 0.35), xytext=(5.2, 0.52),
                fontsize=11, color=COLORS["coral"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=COLORS["coral"], lw=1.2),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                         edgecolor=COLORS["coral"], alpha=0.9))
    ax.set_ylim(0, 0.6)

    # --- Multinomial NB ---
    ax = axes[1]
    words = ["free", "call", "win", "hello", "thanks"]
    ham_probs = [0.02, 0.08, 0.01, 0.15, 0.12]
    spam_probs = [0.18, 0.14, 0.12, 0.03, 0.02]
    x_pos = np.arange(len(words))
    width = 0.35
    ax.bar(x_pos - width/2, ham_probs, width, color=COLORS["blue"], alpha=0.7, label="Ham")
    ax.bar(x_pos + width/2, spam_probs, width, color=COLORS["coral"], alpha=0.7, label="Spam")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(words, fontsize=10)
    ax.set_title("Multinomial NB", fontweight="bold", fontsize=13)
    ax.set_xlabel("Word")
    ax.set_ylabel(r"$P(\mathrm{word} \mid c)$")
    ax.legend(fontsize=9)

    # --- Bernoulli NB ---
    ax = axes[2]
    words_b = ["free", "call", "win", "hello", "thanks"]
    ham_p = [0.05, 0.30, 0.02, 0.55, 0.45]
    spam_p = [0.70, 0.60, 0.50, 0.08, 0.05]
    x_pos = np.arange(len(words_b))
    ax.bar(x_pos - width/2, ham_p, width, color=COLORS["blue"], alpha=0.7, label="Ham")
    ax.bar(x_pos + width/2, spam_p, width, color=COLORS["coral"], alpha=0.7, label="Spam")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(words_b, fontsize=10)
    ax.set_title("Bernoulli NB", fontweight="bold", fontsize=13)
    ax.set_xlabel("Word")
    ax.set_ylabel(r"$P(\mathrm{word\ present} \mid c)$")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.85)
    ax.axhline(0.5, color=COLORS["gray"], linestyle="--", alpha=0.3, linewidth=1)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations", "fig2_nb_variants.png"), dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("  ✓ fig2_nb_variants.png")


# ═══════════════════════════════════════════════════════════════════
# Fig 3: Generative vs Discriminative (conceptual learning curves)
# ═══════════════════════════════════════════════════════════════════
def fig3_generative_vs_discriminative():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Conceptual model comparison ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title("Generative vs Discriminative", fontweight="bold", fontsize=13)

    # Generative side
    gen_box = FancyBboxPatch((0.5, 5.5), 4, 3.5, boxstyle="round,pad=0.3",
                              facecolor=COLORS["lightblue"], edgecolor=COLORS["blue"],
                              linewidth=2)
    ax.add_patch(gen_box)
    ax.text(2.5, 8.3, "Generative", ha="center", va="center",
            fontsize=12, fontweight="bold", color=COLORS["blue"])
    ax.text(2.5, 7.3, r"Models $P(\mathbf{x} \mid c) P(c)$", ha="center", fontsize=10)
    ax.text(2.5, 6.5, "Then applies Bayes' rule\nto get posterior", ha="center",
            fontsize=9, color=COLORS["gray"], style="italic")

    # Discriminative side
    disc_box = FancyBboxPatch((5.5, 5.5), 4, 3.5, boxstyle="round,pad=0.3",
                               facecolor=COLORS["lightcoral"], edgecolor=COLORS["coral"],
                               linewidth=2)
    ax.add_patch(disc_box)
    ax.text(7.5, 8.3, "Discriminative", ha="center", va="center",
            fontsize=12, fontweight="bold", color=COLORS["coral"])
    ax.text(7.5, 7.3, r"Models $P(c \mid \mathbf{x})$ directly", ha="center", fontsize=10)
    ax.text(7.5, 6.5, "Learns decision boundary\nfrom data", ha="center",
            fontsize=9, color=COLORS["gray"], style="italic")

    # Properties below
    props_gen = ["Closed-form training", "Fast with few samples", "Can generate data"]
    props_disc = ["Iterative optimization", "Better with lots of data", "Only classifies"]
    for i, (pg, pd) in enumerate(zip(props_gen, props_disc)):
        y_pos = 4.0 - i * 1.2
        ax.text(2.5, y_pos, f"• {pg}", ha="center", fontsize=9, color=COLORS["blue"])
        ax.text(7.5, y_pos, f"• {pd}", ha="center", fontsize=9, color=COLORS["coral"])

    ax.axis("off")

    # --- Right: Ng & Jordan learning curves (stylized) ---
    ax = axes[1]
    n_samples = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    # Stylized curves matching Ng & Jordan 2002 findings
    nb_error = 0.22 * np.exp(-n_samples / 200) + 0.05
    lr_error = 0.35 * np.exp(-n_samples / 500) + 0.03

    ax.plot(n_samples, nb_error, "o-", color=COLORS["blue"], linewidth=2.5,
            markersize=6, label="Naive Bayes (generative)")
    ax.plot(n_samples, lr_error, "s-", color=COLORS["coral"], linewidth=2.5,
            markersize=6, label="Logistic Regression (discriminative)")

    # Crossover annotation
    cross_idx = np.argmin(np.abs(nb_error - lr_error))
    ax.axvline(n_samples[cross_idx], color=COLORS["gray"], linestyle="--", alpha=0.4)
    ax.annotate("Crossover\npoint", xy=(n_samples[cross_idx], nb_error[cross_idx]),
                xytext=(n_samples[cross_idx] * 2.5, nb_error[cross_idx] + 0.06),
                fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color=COLORS["gray"]),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                         edgecolor=COLORS["gray"], alpha=0.8))

    # Region labels
    ax.fill_betweenx([0, 0.4], 0, n_samples[cross_idx],
                     alpha=0.05, color=COLORS["blue"])
    ax.fill_betweenx([0, 0.4], n_samples[cross_idx], 6000,
                     alpha=0.05, color=COLORS["coral"])
    ax.text(60, 0.30, "NB wins", fontsize=10, color=COLORS["blue"],
            fontweight="bold", alpha=0.6)
    ax.text(2000, 0.30, "LR wins", fontsize=10, color=COLORS["coral"],
            fontweight="bold", alpha=0.6)

    ax.set_xscale("log")
    ax.set_xlabel("Number of Training Samples (log scale)")
    ax.set_ylabel("Test Error Rate")
    ax.set_title("Learning Curves (Ng & Jordan, 2002)", fontweight="bold", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 0.4)
    ax.set_xlim(8, 6000)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations", "fig3_generative_vs_discriminative.png"), dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("  ✓ fig3_generative_vs_discriminative.png")


# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating Naive Bayes illustrations...")
    fig1_naive_independence()
    fig2_nb_variants()
    fig3_generative_vs_discriminative()
    print("Done.")
