#!/usr/bin/env python3
# ============================================================
#  run.py  —  Master entry point for the Hybrid Battleship AI
#
#  COMMANDS:
#    python run.py setup        ← install deps, check hardware
#    python run.py ga           ← Phase 1: GA + PDF bot  (CPU, ~12-24h)
#    python run.py train_rl     ← Phase 2: RL training   (GPU, ~12-24h)
#    python run.py validate     ← Phase 3: Dual scoring  (CPU+GPU, ~2-4h)
#    python run.py visualize    ← Generate all plots
#    python run.py status       ← Show progress of all phases
#    python run.py all          ← Run everything in order
#    python run.py benchmark    ← Quick PDF bot speed test
#
#  STOP/RESUME:
#    Press Ctrl+C at any time — all phases save automatically.
#    Re-run the same command to continue from where you stopped.
#
#  TYPICAL WORKFLOW (GTX 1650, 8GB RAM):
#    1. python run.py setup
#    2. python run.py ga          ← run overnight (~16h)
#    3. python run.py train_rl    ← run second day (~16h) [OPTIONAL]
#    4. python run.py validate    ← run ~3h
#    5. python run.py visualize
#
#  GA + validate alone (no RL) still gives excellent results
#  and completes in ~20 hours total.
# ============================================================

import os
import sys
import json
import time
import subprocess


def banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║        HYBRID BATTLESHIP AI  —  GTX 1650 Edition            ║
║                                                              ║
║  Phase 1 │ Genetic Algorithm  + PDF Bot  (CPU, ~12-24h)     ║
║  Phase 2 │ RL Self-Play Training         (GPU, ~12-24h)     ║
║  Phase 3 │ Dual Validation               (CPU+GPU, ~2-4h)   ║
║                                                              ║
║  Result  │ Top 10,000 hardest-to-sink layouts               ║
╚══════════════════════════════════════════════════════════════╝
""")


# ── setup ────────────────────────────────────────────────────
def cmd_setup():
    print("── Setup ──────────────────────────────────────────────")

    print("\n[Setup] Installing packages...")
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "-r", "requirements.txt", "--quiet"], check=True)

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[Setup] ✓ GPU : {name}  ({vram:.1f} GB VRAM)")
            if vram < 3.5:
                print("[Setup] ⚠ Less than 3.5GB VRAM — reduce RL_BATCH_SIZE in config.py")
        else:
            print("[Setup] ⚠ No CUDA GPU — RL training will be slow (CPU only)")
            print("         GA + PDF bot runs fine on CPU anyway!")
    except ImportError:
        print("[Setup] ✗ PyTorch not installed")

    import psutil, shutil
    ram_gb  = psutil.virtual_memory().total / 1e9
    free_gb = shutil.disk_usage(".").free / 1e9
    print(f"[Setup] RAM  : {ram_gb:.1f} GB")
    print(f"[Setup] Disk : {free_gb:.1f} GB free")

    if ram_gb < 6:
        print("[Setup] ⚠ Low RAM — reduce GA_WORKERS to 2 in config.py")
    if free_gb < 5:
        print("[Setup] ⚠ Very low disk — checkpoints need ~2GB")

    import config
    for d in [config.DATA_DIR, config.CKPT_DIR, config.LOG_DIR, config.TOP_DIR]:
        os.makedirs(d, exist_ok=True)

    # Quick PDF bot speed test
    print("\n[Setup] Testing PDF bot speed...")
    cmd_benchmark(quick=True)

    print("\n[Setup] ✓ Ready! Start with: python run.py ga")


# ── GA ───────────────────────────────────────────────────────
def cmd_ga():
    print("── Phase 1: Genetic Algorithm + PDF Bot ───────────────")
    print("  Breeds 10,000 layouts over 500 generations.")
    print("  PDF bot is the fitness judge — zero training needed.")
    print("  Ctrl+C saves. Re-run to continue.\n")
    from genetic_algorithm import run_ga
    run_ga()


# ── RL training ──────────────────────────────────────────────
def cmd_train_rl():
    print("── Phase 2: RL Model Training ─────────────────────────")
    print("  Self-play with PDF-bot warmup. Optional but improves")
    print("  final validation quality.")
    print("  Ctrl+C saves. Re-run to continue.\n")
    from train_rl import train_rl
    train_rl()


# ── Validate ─────────────────────────────────────────────────
def cmd_validate():
    print("── Phase 3: Dual Validation ────────────────────────────")
    print("  Re-scores GA winners against both PDF bot and RL model.")
    print("  Ctrl+C saves. Re-run to continue.\n")
    from validate import validate
    validate()


# ── Visualize ────────────────────────────────────────────────
def cmd_visualize():
    print("── Visualize ───────────────────────────────────────────")
    from visualize import (plot_ga_curve, plot_rl_curve,
                           show_top_layouts, layout_heatmap, print_summary)
    print_summary()
    plot_ga_curve()
    plot_rl_curve()
    show_top_layouts(n=20)
    layout_heatmap()
    import config
    print(f"\n  Images saved to:")
    print(f"    {config.LOG_DIR}/ga_curve.png")
    print(f"    {config.LOG_DIR}/rl_curve.png")
    print(f"    {config.TOP_DIR}/top_layouts.png")
    print(f"    {config.TOP_DIR}/heatmap.png")


# ── Status ───────────────────────────────────────────────────
def cmd_status():
    import config
    import numpy as np

    print("── Status ──────────────────────────────────────────────\n")

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU      : {torch.cuda.get_device_name(0)}")
        else:
            print("GPU      : Not available (CPU mode)")
    except:
        print("GPU      : PyTorch not found")

    # Phase 1 - GA
    if os.path.exists(config.GA_CKPT_FILE):
        with open(config.GA_CKPT_FILE) as f:
            state = json.load(f)
        gen  = state['generation']
        best = state['best_score']
        mean = state['mean_score']
        pct  = gen / config.GA_GENERATIONS * 100
        elapsed_h = state.get('elapsed_time', 0) / 3600
        print(f"GA       : Gen {gen}/{config.GA_GENERATIONS} ({pct:.0f}%) | "
              f"best={best:.1f} | mean={mean:.1f} | {elapsed_h:.1f}h elapsed")
    else:
        print("GA       : Not started")

    # Phase 2 - RL
    if os.path.exists(config.RL_TRAIN_FILE):
        with open(config.RL_TRAIN_FILE) as f:
            state = json.load(f)
        ep  = state['episode']
        pct = ep / config.RL_TOTAL_EPISODES * 100
        print(f"RL train : {ep:,}/{config.RL_TOTAL_EPISODES:,} eps ({pct:.0f}%)")
    else:
        print("RL train : Not started  (optional)")

    # Phase 3 - Validation
    val_progress = os.path.join(config.CKPT_DIR, "val_progress.json")
    if os.path.exists(config.TOP_LAYOUTS_FILE):
        layouts = np.load(config.TOP_LAYOUTS_FILE)
        scores  = np.load(config.TOP_SCORES_FILE)
        print(f"Validate : ✓ Complete — {len(layouts):,} layouts | "
              f"best combined={scores[0]:.4f}")
    elif os.path.exists(val_progress):
        with open(val_progress) as f:
            vp = json.load(f)
        print(f"Validate : In progress — {vp['n_processed']:,} scored")
    else:
        print("Validate : Not started")

    # GA intermediate results
    ga_top = os.path.join(config.CKPT_DIR, "ga_top_layouts.npy")
    if os.path.exists(ga_top):
        layouts = np.load(ga_top)
        ga_scores = np.load(os.path.join(config.CKPT_DIR, "ga_top_scores.npy"))
        print(f"GA output: {len(layouts):,} layouts | "
              f"best PDF score={ga_scores[0]:.1f}")

    import shutil
    print(f"\nDisk free: {shutil.disk_usage('.').free/1e9:.1f} GB")
    print("\nNext step:")
    if not os.path.exists(config.GA_CKPT_FILE):
        print("  python run.py ga")
    elif not os.path.exists(ga_top):
        print("  python run.py ga  ← GA still running")
    elif not os.path.exists(config.TOP_LAYOUTS_FILE):
        print("  python run.py validate  (or train_rl first for better results)")
    else:
        print("  python run.py visualize  ← you're done!")


# ── Benchmark ────────────────────────────────────────────────
def cmd_benchmark(quick=False):
    import time, numpy as np
    from bitboard import random_layout
    from pdf_bot import score_layout_pdf

    n_layouts = 5 if quick else 20
    n_games   = 20 if quick else 100

    print(f"  Running {n_layouts} layouts × {n_games} games each...")
    scores = []
    t0     = time.time()
    for i in range(n_layouts):
        bits, placements = random_layout()
        s = score_layout_pdf(bits, placements, n_games=n_games)
        scores.append(s)
        if not quick:
            print(f"  Layout {i+1:>2}: {s:.1f} shots survived")

    elapsed = time.time() - t0
    total_games = n_layouts * n_games
    gps  = total_games / elapsed
    print(f"\n  Avg score    : {np.mean(scores):.1f} shots")
    print(f"  Total time   : {elapsed:.1f}s")
    print(f"  Speed        : {gps:.0f} games/sec")

    # Estimate GA time
    ga_games = config.GA_POPULATION * config.GA_FITNESS_GAMES * config.GA_GENERATIONS
    est_h    = ga_games / gps / config.GA_WORKERS / 3600
    print(f"\n  Estimated GA time ({config.GA_GENERATIONS} gens, "
          f"{config.GA_WORKERS} workers): {est_h:.1f}h")


# ── All ──────────────────────────────────────────────────────
def cmd_all():
    banner()
    print("Running full pipeline...\n")
    cmd_ga()
    print("\n" + "─"*60)
    cmd_train_rl()
    print("\n" + "─"*60)
    cmd_validate()
    print("\n" + "─"*60)
    cmd_visualize()
    print("\n══ ALL DONE ══")
    print(f"→ Top layouts: top_layouts/top_10000.npy")
    print(f"→ Visuals    : top_layouts/top_layouts.png")


# ── help ─────────────────────────────────────────────────────
def cmd_help():
    print("""
Commands:
  python run.py setup       Check hardware, install deps, speed-test PDF bot
  python run.py ga          Phase 1: Genetic Algorithm (CPU) — main optimizer
  python run.py train_rl    Phase 2: RL training (GPU) — optional enhancer
  python run.py validate    Phase 3: Score GA results with both attackers
  python run.py visualize   Generate plots and images
  python run.py status      Show progress of each phase
  python run.py benchmark   Test PDF bot speed on your machine
  python run.py all         Run all phases automatically

Recommended order for 24-hour result (GA only):
  python run.py setup
  python run.py ga          ← ~16 hours on your CPU
  python run.py validate    ← ~3 hours (PDF-only, no RL needed)
  python run.py visualize

Full hybrid (48-hour, best quality):
  python run.py ga          ← Phase 1 overnight
  python run.py train_rl    ← Phase 2 next day
  python run.py validate    ← Phase 3
  python run.py visualize

Stop any phase with Ctrl+C — it saves and you can resume later.

Output files:
  top_layouts/top_10000.npy         ← final top 10K layouts (shape: 10000,10,10)
  top_layouts/top_10000_scores.npy  ← combined score per layout
  top_layouts/top_10000_meta.json   ← stats about the run
  top_layouts/top_layouts.png       ← visual grid of top 20
  top_layouts/heatmap.png           ← cell occupation frequency
  logs/ga_curve.png                 ← GA convergence over generations
  logs/rl_curve.png                 ← RL training loss & turns
""")


# ── Entry ────────────────────────────────────────────────────
COMMANDS = {
    "setup":     cmd_setup,
    "ga":        cmd_ga,
    "train_rl":  cmd_train_rl,
    "validate":  cmd_validate,
    "visualize": cmd_visualize,
    "status":    cmd_status,
    "benchmark": cmd_benchmark,
    "all":       cmd_all,
    "help":      cmd_help,
    "--help":    cmd_help,
    "-h":        cmd_help,
}

if __name__ == "__main__":
    banner()
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else "help"
    if cmd not in COMMANDS:
        print(f"Unknown command: '{cmd}'")
        cmd_help()
        sys.exit(1)
    COMMANDS[cmd]()
