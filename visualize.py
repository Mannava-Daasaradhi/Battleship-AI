# ============================================================
#  visualize.py
#  Show top layouts, training curves, and layout statistics
# ============================================================

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import config


def plot_layout(layout, ax, title="", score=None):
    """Plot a single 10x10 layout on given axes."""
    cmap = ListedColormap(['#1a3a5c', '#e8b84b'])  # ocean blue, ship gold
    ax.imshow(layout, cmap=cmap, vmin=0, vmax=1, aspect='equal')
    ax.set_xticks(range(config.BOARD_SIZE))
    ax.set_yticks(range(config.BOARD_SIZE))
    ax.set_xticklabels([chr(65+i) for i in range(config.BOARD_SIZE)], fontsize=7)
    ax.set_yticklabels(range(1, config.BOARD_SIZE+1), fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor('#334455')

    label = title
    if score is not None:
        label += f"\n{score:.1f} turns"
    ax.set_title(label, fontsize=9, pad=4)


def show_top_layouts(n=20, save_path=None):
    """Plot the top N layouts from the scoring results."""
    layouts_path = config.TOP_LAYOUTS_FILE
    scores_path  = os.path.join(config.TOP_DIR, "top_10000_scores.npy")

    if not os.path.exists(layouts_path):
        print("[Viz] Top layouts not found. Run: python run.py score")
        return

    layouts = np.load(layouts_path)[:n]
    scores  = np.load(scores_path)[:n] if os.path.exists(scores_path) else [None]*n

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.8))
    fig.suptitle("Top Battleship Layouts", fontsize=14, fontweight='bold', y=1.01)
    fig.patch.set_facecolor('#0d1f2d')

    axes_flat = axes.flatten() if n > 1 else [axes]
    for i, ax in enumerate(axes_flat):
        if i < n:
            plot_layout(layouts[i], ax, title=f"#{i+1}", score=scores[i])
        else:
            ax.axis('off')

    plt.tight_layout()
    out = save_path or os.path.join(config.TOP_DIR, "top_layouts.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1f2d')
    print(f"[Viz] Saved: {out}")
    plt.close()


def plot_training_curve(save_path=None):
    """Plot avg turns and loss over training."""
    log_path = os.path.join(config.LOG_DIR, "train_log.csv")
    if not os.path.exists(log_path):
        print("[Viz] Training log not found.")
        return

    data = np.genfromtxt(log_path, delimiter=',', names=True, skip_header=0)
    if data.size == 0:
        print("[Viz] Training log is empty.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), facecolor='#0d1f2d')

    # Smooth with rolling average
    def smooth(x, w=50):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w)/w, mode='valid')

    eps = data['episode']

    # Turns plot
    ax1.set_facecolor('#1a2a3a')
    ax1.plot(eps[:len(smooth(data['avg_turns']))],
             smooth(data['avg_turns']), color='#e8b84b', linewidth=1.5)
    ax1.set_ylabel("Avg Turns to Win", color='white')
    ax1.set_title("Training Progress", color='white', fontsize=13)
    ax1.tick_params(colors='white')
    ax1.set_xlabel("")
    for spine in ax1.spines.values():
        spine.set_edgecolor('#334455')

    # Loss plot
    ax2.set_facecolor('#1a2a3a')
    ax2.plot(eps[:len(smooth(data['avg_loss']))],
             smooth(data['avg_loss']), color='#5bc8af', linewidth=1.5)
    ax2.set_ylabel("Training Loss", color='white')
    ax2.set_xlabel("Episode", color='white')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#334455')

    plt.tight_layout()
    out = save_path or os.path.join(config.LOG_DIR, "training_curve.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1f2d')
    print(f"[Viz] Saved: {out}")
    plt.close()


def layout_heatmap(save_path=None):
    """Show heatmap of how often each cell is occupied in top 10K layouts."""
    layouts_path = config.TOP_LAYOUTS_FILE
    if not os.path.exists(layouts_path):
        print("[Viz] Top layouts not found.")
        return

    layouts = np.load(layouts_path)
    heatmap = layouts.mean(axis=0)  # (10,10) fraction occupied

    fig, ax = plt.subplots(figsize=(6, 5.5), facecolor='#0d1f2d')
    im = ax.imshow(heatmap, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Fraction occupied')
    ax.set_title("Cell Occupation Heatmap\n(Top 10,000 layouts)", color='white', fontsize=12)
    ax.set_xticks(range(config.BOARD_SIZE))
    ax.set_xticklabels([chr(65+i) for i in range(config.BOARD_SIZE)], color='white')
    ax.set_yticks(range(config.BOARD_SIZE))
    ax.set_yticklabels(range(1, config.BOARD_SIZE+1), color='white')

    out = save_path or os.path.join(config.TOP_DIR, "heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1f2d')
    print(f"[Viz] Saved: {out}")
    plt.close()


def print_layout_ascii(layout, idx=0, score=None):
    """Print a single layout to terminal."""
    cols = "  A B C D E F G H I J"
    sep  = "  " + "─" * 19
    print(f"\n  Layout #{idx+1}" + (f"  (score: {score:.1f})" if score else ""))
    print(cols)
    print(sep)
    for r in range(config.BOARD_SIZE):
        row_str = f"{r+1:>2}│"
        for c in range(config.BOARD_SIZE):
            row_str += "█ " if layout[r, c] == 1 else "· "
        print(row_str)


if __name__ == "__main__":
    plot_training_curve()
    show_top_layouts(n=20)
    layout_heatmap()
    print("[Viz] Done!")
