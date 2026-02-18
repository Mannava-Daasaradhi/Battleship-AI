# ============================================================
#  visualize.py
#  Generate all plots and result images.
# ============================================================

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import config

BOARD = config.BOARD_SIZE
OCEAN = '#0a1628'
SHIP  = '#f0a500'
GRID  = '#1e3a5f'


# ── Single layout renderer ───────────────────────────────────
def plot_layout(layout, ax, title="", score=None, rank=None):
    cmap = ListedColormap([OCEAN, SHIP])
    ax.imshow(layout, cmap=cmap, vmin=0, vmax=1, aspect='equal',
              interpolation='nearest')

    # Grid lines
    for i in range(BOARD + 1):
        ax.axhline(i - 0.5, color=GRID, linewidth=0.5)
        ax.axvline(i - 0.5, color=GRID, linewidth=0.5)

    ax.set_xticks(range(BOARD))
    ax.set_yticks(range(BOARD))
    ax.set_xticklabels([chr(65+i) for i in range(BOARD)], fontsize=6, color='#aabbcc')
    ax.set_yticklabels(range(1, BOARD+1), fontsize=6, color='#aabbcc')
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    label = ""
    if rank is not None:
        label += f"#{rank}  "
    if score is not None:
        label += f"{score:.1f}shots"
    if title:
        label = title + ("\n" + label if label else "")
    ax.set_title(label, fontsize=8, pad=3, color='#ddeeff')


def show_top_layouts(n=20, save_path=None):
    if not os.path.exists(config.TOP_LAYOUTS_FILE):
        print("[Viz] Top layouts not found. Run: python run.py validate")
        return

    layouts = np.load(config.TOP_LAYOUTS_FILE)[:n]
    scores  = np.load(config.TOP_SCORES_FILE)[:n] if os.path.exists(config.TOP_SCORES_FILE) else [None]*n
    meta    = {}
    if os.path.exists(config.TOP_META_FILE):
        with open(config.TOP_META_FILE) as f:
            meta = json.load(f)

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 2.4, rows * 2.6),
                              facecolor=OCEAN)
    fig.suptitle(f"Top {n} Battleship Layouts\n"
                 f"PDF {meta.get('pdf_games','?')} games + "
                 f"RL {meta.get('rl_games','?')} games | "
                 f"Weight {meta.get('pdf_weight','?')}/{meta.get('rl_weight','?')}",
                 fontsize=10, color='white', y=1.01)

    axes_flat = axes.flatten() if n > 1 else [axes]
    for i, ax in enumerate(axes_flat):
        ax.set_facecolor(OCEAN)
        if i < n:
            plot_layout(layouts[i], ax, score=scores[i], rank=i+1)
        else:
            ax.axis('off')

    plt.tight_layout()
    out = save_path or os.path.join(config.TOP_DIR, "top_layouts.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=OCEAN)
    print(f"[Viz] Saved: {out}")
    plt.close()


def layout_heatmap(save_path=None):
    """Show which cells the top layouts occupy most often."""
    if not os.path.exists(config.TOP_LAYOUTS_FILE):
        print("[Viz] Top layouts not found.")
        return

    layouts = np.load(config.TOP_LAYOUTS_FILE)
    heatmap = layouts.mean(axis=0)

    fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=OCEAN)
    ax.set_facecolor(OCEAN)
    im = ax.imshow(heatmap, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fraction occupied', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    ax.set_title(f"Cell Occupation Heatmap\n(Top {len(layouts):,} layouts)",
                 color='white', fontsize=12)
    ax.set_xticks(range(BOARD))
    ax.set_xticklabels([chr(65+i) for i in range(BOARD)], color='white')
    ax.set_yticks(range(BOARD))
    ax.set_yticklabels(range(1, BOARD+1), color='white')

    out = save_path or os.path.join(config.TOP_DIR, "heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=OCEAN)
    print(f"[Viz] Saved: {out}")
    plt.close()


def plot_ga_curve(save_path=None):
    log_path = os.path.join(config.LOG_DIR, "ga_log.csv")
    if not os.path.exists(log_path):
        print("[Viz] GA log not found.")
        return

    data = np.genfromtxt(log_path, delimiter=',', names=True)
    if data.size == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=OCEAN)
    ax.set_facecolor('#0d1f35')

    ax.plot(data['generation'], data['best_score'],
            color=SHIP,     linewidth=2,   label='Best layout score')
    ax.plot(data['generation'], data['mean_score'],
            color='#5bc8af', linewidth=1.5, label='Mean population score')
    ax.plot(data['generation'], data['top10_mean'],
            color='#8899ff', linewidth=1.5, linestyle='--', label='Top-10 mean score')

    ax.set_xlabel('Generation', color='white')
    ax.set_ylabel('Avg shots survived (vs PDF bot)', color='white')
    ax.set_title('Genetic Algorithm Convergence', color='white', fontsize=13)
    ax.legend(facecolor='#1a2a3a', labelcolor='white', fontsize=9)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    out = save_path or os.path.join(config.LOG_DIR, "ga_curve.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=OCEAN)
    print(f"[Viz] Saved: {out}")
    plt.close()


def plot_rl_curve(save_path=None):
    log_path = os.path.join(config.LOG_DIR, "rl_log.csv")
    if not os.path.exists(log_path):
        print("[Viz] RL log not found. (RL training may not have run yet)")
        return

    data = np.genfromtxt(log_path, delimiter=',', names=True)
    if data.size == 0:
        return

    def smooth(x, w=50):
        if len(x) < w: return x
        return np.convolve(x, np.ones(w)/w, mode='valid')

    eps = data['episode']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), facecolor=OCEAN)

    ax1.set_facecolor('#0d1f35')
    ax1.plot(eps[:len(smooth(data['avg_turns']))],
             smooth(data['avg_turns']), color=SHIP, linewidth=1.5)
    ax1.set_ylabel('Avg turns to win', color='white')
    ax1.set_title('RL Training Progress', color='white', fontsize=13)
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values(): spine.set_edgecolor(GRID)

    ax2.set_facecolor('#0d1f35')
    ax2.plot(eps[:len(smooth(data['avg_loss']))],
             smooth(data['avg_loss']), color='#5bc8af', linewidth=1.5)
    ax2.set_ylabel('Training loss', color='white')
    ax2.set_xlabel('Episode', color='white')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values(): spine.set_edgecolor(GRID)

    out = save_path or os.path.join(config.LOG_DIR, "rl_curve.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=OCEAN)
    print(f"[Viz] Saved: {out}")
    plt.close()


def print_layout_ascii(layout, rank=0, pdf_score=None, combined=None):
    cols = "  A B C D E F G H I J"
    sep  = "  " + "─" * 19
    print(f"\n  Layout #{rank}" +
          (f"  PDF={pdf_score:.1f}" if pdf_score else "") +
          (f"  combined={combined:.4f}" if combined else ""))
    print(cols)
    print(sep)
    for r in range(BOARD):
        row_str = f"{r+1:>2}│"
        for c in range(BOARD):
            row_str += "█ " if layout[r, c] == 1 else "· "
        print(row_str)


def print_summary():
    """Print text summary of results."""
    if not os.path.exists(config.TOP_META_FILE):
        print("[Viz] No results yet. Run the full pipeline first.")
        return

    with open(config.TOP_META_FILE) as f:
        meta = json.load(f)

    print("\n" + "═"*50)
    print("  HYBRID BATTLESHIP AI — FINAL RESULTS")
    print("═"*50)
    print(f"  Total top layouts     : {meta['n_layouts']:,}")
    print(f"  PDF games per layout  : {meta['pdf_games']}")
    print(f"  RL games per layout   : {meta['rl_games']}")
    print(f"  RL model used         : {meta['rl_model_used']}")
    print(f"  Best PDF score        : {meta['best_pdf_score']:.2f} shots survived")
    print(f"  Best RL score         : {meta['best_rl_score']:.2f} shots survived")
    print(f"  Mean PDF (top 100)    : {meta['mean_pdf_top100']:.2f} shots survived")
    print(f"  Mean PDF (all 10K)    : {meta['mean_pdf_all']:.2f} shots survived")
    print("═"*50)

    if os.path.exists(config.TOP_LAYOUTS_FILE):
        layouts = np.load(config.TOP_LAYOUTS_FILE)
        scores  = np.load(config.TOP_SCORES_FILE)
        print(f"\n  Top 3 layouts:\n")
        for i in range(min(3, len(layouts))):
            print_layout_ascii(layouts[i], rank=i+1, combined=scores[i])


if __name__ == "__main__":
    print_summary()
    plot_ga_curve()
    plot_rl_curve()
    show_top_layouts(n=20)
    layout_heatmap()
    print("\n[Viz] All done!")
