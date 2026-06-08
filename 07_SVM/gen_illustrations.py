"""Generate conceptual illustrations for the Support Vector Machines notebook."""

import os
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations"), exist_ok=True)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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
# Fig 1: Maximum Margin Concept
# ═══════════════════════════════════════════════════════════════════
def fig1_maximum_margin():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    np.random.seed(42)
    n = 20
    class1 = np.random.randn(n, 2) * 0.6 + np.array([-1.5, -1.0])
    class2 = np.random.randn(n, 2) * 0.6 + np.array([1.5, 1.0])
    x_line = np.linspace(-3.5, 3.5, 100)

    # ---- Left: Multiple separators, only one is optimal ----
    ax = axes[0]
    ax.scatter(class1[:, 0], class1[:, 1], c=COLORS["coral"], s=50, zorder=3,
               edgecolors="white", linewidths=0.5, label="Class $-1$")
    ax.scatter(class2[:, 0], class2[:, 1], c=COLORS["blue"], s=50, zorder=3,
               edgecolors="white", linewidths=0.5, label="Class $+1$")

    for slope, intercept in [(-0.6, 0.3), (-1.2, 0.1), (-0.9, -0.5)]:
        ax.plot(x_line, slope * x_line + intercept, color=COLORS["gray"],
                linewidth=1, linestyle="--", alpha=0.5)

    ax.plot(x_line, -x_line * 0.85 + 0.05, color="black", linewidth=2.5, label="Max margin")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3, 3)
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_title("Many Valid Separators, One Optimal", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")

    # ---- Right: Margin geometry ----
    ax = axes[1]
    ax.scatter(class1[:, 0], class1[:, 1], c=COLORS["coral"], s=50, zorder=3,
               edgecolors="white", linewidths=0.5)
    ax.scatter(class2[:, 0], class2[:, 1], c=COLORS["blue"], s=50, zorder=3,
               edgecolors="white", linewidths=0.5)

    slope_opt, intercept_opt = -0.85, 0.05
    margin_offset = 0.85
    ax.plot(x_line, slope_opt * x_line + intercept_opt, color="black", linewidth=2.5)
    ax.plot(x_line, slope_opt * x_line + intercept_opt + margin_offset,
            color="black", linewidth=1.5, linestyle="--")
    ax.plot(x_line, slope_opt * x_line + intercept_opt - margin_offset,
            color="black", linewidth=1.5, linestyle="--")
    ax.fill_between(x_line,
                    slope_opt * x_line + intercept_opt - margin_offset,
                    slope_opt * x_line + intercept_opt + margin_offset,
                    alpha=0.08, color=COLORS["green"])

    ax.annotate(
        "", xy=(0.4, slope_opt * 0.4 + intercept_opt + margin_offset),
        xytext=(0.4, slope_opt * 0.4 + intercept_opt - margin_offset),
        arrowprops=dict(arrowstyle="<->", color=COLORS["green"], lw=2),
    )
    ax.text(0.7, slope_opt * 0.4 + intercept_opt, r"Margin $= \frac{2}{\|\mathbf{w}\|}$",
            fontsize=11, color=COLORS["green"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["green"], alpha=0.9))

    for pts in [class1, class2]:
        dists = np.abs(slope_opt * pts[:, 0] - pts[:, 1] + intercept_opt) / np.sqrt(slope_opt**2 + 1)
        sv_idx = np.argsort(dists)[:2]
        ax.scatter(pts[sv_idx, 0], pts[sv_idx, 1], facecolors="none",
                   edgecolors="black", s=200, linewidths=2.5, zorder=4)

    ax.text(-3.2, 2.5, "Support vectors", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9))
    ax.annotate("", xy=(-1.8, -0.2), xytext=(-2.5, 2.2),
                arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5))

    mid_x, mid_y = 0.0, intercept_opt
    norm = np.sqrt(1 + slope_opt**2)
    wx, wy = 1 / norm, -slope_opt / norm
    ax.annotate("", xy=(mid_x + wx * 1.2, mid_y + wy * 1.2), xytext=(mid_x, mid_y),
                arrowprops=dict(arrowstyle="-|>", color=COLORS["purple"], lw=2.5))
    ax.text(mid_x + wx * 1.2 + 0.1, mid_y + wy * 1.2 + 0.15,
            r"$\mathbf{w}$", fontsize=14, color=COLORS["purple"], fontweight="bold")

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3, 3)
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_title("Maximum Margin Geometry", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations", "fig1_maximum_margin.png"), bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved fig1_maximum_margin.png")


# ═══════════════════════════════════════════════════════════════════
# Fig 2: The Kernel Trick
# ═══════════════════════════════════════════════════════════════════
def fig2_kernel_trick():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    np.random.seed(42)

    # ---- Left: Non-separable concentric rings ----
    ax = axes[0]
    theta_in = np.random.uniform(0, 2 * np.pi, 40)
    r_in = np.random.normal(0.8, 0.15, 40)
    x_in, y_in = r_in * np.cos(theta_in), r_in * np.sin(theta_in)

    theta_out = np.random.uniform(0, 2 * np.pi, 60)
    r_out = np.random.normal(2.0, 0.2, 60)
    x_out, y_out = r_out * np.cos(theta_out), r_out * np.sin(theta_out)

    ax.scatter(x_in, y_in, c=COLORS["coral"], s=30, zorder=3,
               edgecolors="white", linewidths=0.5, label="Class $-1$")
    ax.scatter(x_out, y_out, c=COLORS["blue"], s=30, zorder=3,
               edgecolors="white", linewidths=0.5, label="Class $+1$")
    ax.plot([-3, 3], [0.5, -0.5], color=COLORS["gray"], linewidth=1.5,
            linestyle="--", alpha=0.5)
    ax.text(1.8, -2.0, "No linear\nseparator!", fontsize=10, ha="center",
            color=COLORS["coral"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["coral"], alpha=0.9))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.set_title("Original Space (not separable)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")

    # ---- Right: After mapping ----
    ax = axes[1]
    r2_in = x_in**2 + y_in**2
    r2_out = x_out**2 + y_out**2

    ax.scatter(np.sqrt(r2_in), r2_in, c=COLORS["coral"], s=30, zorder=3,
               edgecolors="white", linewidths=0.5, label="Class $-1$")
    ax.scatter(np.sqrt(r2_out), r2_out, c=COLORS["blue"], s=30, zorder=3,
               edgecolors="white", linewidths=0.5, label="Class $+1$")

    sep_val = 2.0
    ax.axhline(sep_val, color="black", linewidth=2.5, label="Linear separator")
    ax.axhline(sep_val - 0.5, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.axhline(sep_val + 0.5, color="black", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.fill_between([0, 3.5], sep_val - 0.5, sep_val + 0.5, alpha=0.08, color=COLORS["green"])
    ax.text(2.8, 3.2, "Linearly\nseparable!", fontsize=10, ha="center",
            color=COLORS["green"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLORS["green"], alpha=0.9))
    ax.set_xlim(0, 3.5)
    ax.set_ylim(-0.5, 6)
    ax.set_xlabel(r"$\|\mathbf{x}\|$", fontsize=12)
    ax.set_ylabel(r"$\|\mathbf{x}\|^2 = \phi(\mathbf{x})$", fontsize=12)
    ax.set_title(r"Feature Space via $\phi$  (separable)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    fig.text(0.50, 0.5, r"$\phi(\mathbf{x})$", fontsize=16, ha="center", va="center",
             fontweight="bold", color=COLORS["purple"],
             bbox=dict(boxstyle="rarrow,pad=0.3", fc=COLORS["lightorange"],
                       ec=COLORS["purple"], lw=2))

    plt.tight_layout(w_pad=4)
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations", "fig2_kernel_trick.png"), bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved fig2_kernel_trick.png")


# ═══════════════════════════════════════════════════════════════════
# Fig 3: Soft Margin / Effect of C
# ═══════════════════════════════════════════════════════════════════
def fig3_soft_margin_c():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    np.random.seed(42)

    n = 30
    class1 = np.random.randn(n, 2) * 0.7 + np.array([-1.2, -0.8])
    class2 = np.random.randn(n, 2) * 0.7 + np.array([1.2, 0.8])
    class1 = np.vstack([class1, np.array([[0.5, 0.3], [0.8, -0.2]])])
    class2 = np.vstack([class2, np.array([[-0.3, -0.5], [-0.6, 0.2]])])

    configs = [
        ("Small C (wide margin)", 0.3, COLORS["blue"]),
        ("Medium C (balanced)", 1.5, COLORS["green"]),
        ("Large C (narrow margin)", 20.0, COLORS["coral"]),
    ]

    for idx, (title, margin_width_factor, color) in enumerate(configs):
        ax = axes[idx]
        ax.scatter(class1[:, 0], class1[:, 1], c=COLORS["coral"], s=40, zorder=3,
                   edgecolors="white", linewidths=0.5)
        ax.scatter(class2[:, 0], class2[:, 1], c=COLORS["blue"], s=40, zorder=3,
                   edgecolors="white", linewidths=0.5)

        x_line = np.linspace(-3.5, 3.5, 100)
        slope, intercept = -0.85, 0.05
        margin = 1.8 / margin_width_factor

        ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=2)
        ax.plot(x_line, slope * x_line + intercept + margin,
                color="black", linewidth=1.2, linestyle="--")
        ax.plot(x_line, slope * x_line + intercept - margin,
                color="black", linewidth=1.2, linestyle="--")
        ax.fill_between(x_line,
                        slope * x_line + intercept - margin,
                        slope * x_line + intercept + margin,
                        alpha=0.08, color=color)

        if margin_width_factor < 1.0:
            for pt in class1[-2:]:
                if pt[1] > slope * pt[0] + intercept - margin:
                    ax.plot([pt[0], pt[0]],
                            [pt[1], slope * pt[0] + intercept - margin],
                            color=COLORS["orange"], linewidth=1.5, linestyle=":")
                    ax.text(pt[0] + 0.15, (pt[1] + slope * pt[0] + intercept - margin) / 2,
                            r"$\xi_i$", fontsize=9, color=COLORS["orange"])

        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel(r"$x_1$", fontsize=11)
        if idx == 0:
            ax.set_ylabel(r"$x_2$", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", color=color)

    axes[0].text(0, -4.3, "High bias, low variance\nAllows misclassifications",
                 fontsize=9, ha="center", color=COLORS["gray"], style="italic")
    axes[2].text(0, -4.3, "Low bias, high variance\nOverfits to noise",
                 fontsize=9, ha="center", color=COLORS["gray"], style="italic")

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "illustrations", "fig3_soft_margin_c.png"), bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved fig3_soft_margin_c.png")


if __name__ == "__main__":
    fig1_maximum_margin()
    fig2_kernel_trick()
    fig3_soft_margin_c()
    print("All illustrations generated.")
