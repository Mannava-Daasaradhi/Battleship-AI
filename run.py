#!/usr/bin/env python3
# ============================================================
#  run.py  —  Master entry point
#
#  Usage:
#    python run.py setup        ← install deps, check GPU
#    python run.py generate     ← generate all 15B layouts (optional)
#    python run.py train        ← train the AI (resume-safe)
#    python run.py score        ← find top 10K layouts (resume-safe)
#    python run.py visualize    ← plot results
#    python run.py status       ← show current progress
#    python run.py all          ← run everything in order
#
#  STOP ANYTIME with Ctrl+C — it saves and resumes cleanly.
# ============================================================

import os
import sys
import json
import time
import subprocess


def banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║           BATTLESHIP AI  —  GTX 1650 Optimized          ║
║    Train → Score → Find Top 10,000 Defensive Layouts    ║
╚══════════════════════════════════════════════════════════╝
""")


# ── Setup ────────────────────────────────────────────────────
def cmd_setup():
    print("── Step 0: Setup ──────────────────────────────────────")

    # Install requirements
    print("\n[Setup] Installing Python packages...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
                    "--quiet"], check=True)

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            name  = torch.cuda.get_device_name(0)
            vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[Setup] ✓ GPU found: {name}  ({vram:.1f} GB VRAM)")
        else:
            print("[Setup] ⚠ No CUDA GPU found — will run on CPU (much slower)")
    except ImportError:
        print("[Setup] ✗ PyTorch not installed properly")
        return

    # Create directories
    import config
    for d in [config.DATA_DIR, config.CKPT_DIR, config.LOG_DIR, config.TOP_DIR]:
        os.makedirs(d, exist_ok=True)

    # Disk space check
    import shutil
    free_gb = shutil.disk_usage(".").free / 1e9
    print(f"[Setup] Disk free: {free_gb:.1f} GB  (need ~300 GB for all layouts)")
    if free_gb < 50:
        print("[Setup] ⚠ Less than 50GB free — skip layout generation, use on-the-fly sampling")
    elif free_gb < 300:
        print("[Setup] ⚠ Less than 300GB — will use on-the-fly layout sampling during scoring")

    # RAM check
    import psutil
    ram_gb = psutil.virtual_memory().total / 1e9
    print(f"[Setup] RAM: {ram_gb:.1f} GB")

    print("\n[Setup] ✓ All done! Next: python run.py train")


# ── Status ───────────────────────────────────────────────────
def cmd_status():
    import config

    print("── Status ─────────────────────────────────────────────\n")

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.memory_reserved(0) / 1e9
            print(f"GPU : {torch.cuda.get_device_name(0)}")
        else:
            print("GPU : Not available (CPU mode)")
    except:
        print("GPU : PyTorch not installed")

    # Layouts
    if os.path.exists(config.LAYOUTS_FILE):
        from layout_generator import count_layouts
        n = count_layouts()
        print(f"Layouts  : {n/1e9:.3f}B generated  ({os.path.getsize(config.LAYOUTS_FILE)/1e9:.1f} GB)")
    else:
        print("Layouts  : Not generated yet")

    # Training
    if os.path.exists(config.TRAIN_CKPT_FILE):
        with open(config.TRAIN_CKPT_FILE) as f:
            state = json.load(f)
        ep    = state.get('episode', 0)
        turns = state.get('stats', {}).get('avg_turns', '?')
        pct   = ep / config.TOTAL_EPISODES * 100
        print(f"Training : {ep:,} / {config.TOTAL_EPISODES:,} episodes  ({pct:.1f}%)  avg_turns={turns}")
    else:
        print("Training : Not started")

    # Scoring
    if os.path.exists(config.SCORE_CKPT_FILE):
        with open(config.SCORE_CKPT_FILE) as f:
            state = json.load(f)
        r = state.get('round_complete', 0)
        print(f"Scoring  : Round {r}/3 complete")
    else:
        print("Scoring  : Not started")

    # Top layouts
    if os.path.exists(config.TOP_LAYOUTS_FILE):
        layouts = __import__('numpy').load(config.TOP_LAYOUTS_FILE)
        print(f"Top 10K  : ✓ Saved  ({len(layouts):,} layouts)")
    else:
        print("Top 10K  : Not ready yet")

    # Disk
    import shutil
    free = shutil.disk_usage(".").free / 1e9
    print(f"\nDisk free: {free:.1f} GB")
    print("\nRun 'python run.py all' to execute remaining steps automatically.")


# ── Generate layouts ─────────────────────────────────────────
def cmd_generate():
    import shutil
    free_gb = shutil.disk_usage(".").free / 1e9

    print("── Step 1: Generate Layouts ───────────────────────────")
    print(f"  Disk free: {free_gb:.1f} GB")
    print(f"  Layouts file will be ~195 GB")

    if free_gb < 220:
        print("\n  ⚠ Not enough disk space for full generation.")
        print("  The scoring step will sample layouts on-the-fly instead.")
        print("  This is fine — just skip to: python run.py train")
        return

    import config
    from layout_generator import LayoutGenerator
    gen = LayoutGenerator()
    total = gen.generate_all(config.LAYOUTS_FILE)
    print(f"\n✓ Generated {total/1e9:.3f}B layouts")


# ── Train ────────────────────────────────────────────────────
def cmd_train():
    print("── Step 2: Train AI ────────────────────────────────────")
    print("  Press Ctrl+C at any time to save and stop.")
    print("  Re-run this command to resume from last checkpoint.\n")
    from train import train
    train()


# ── Score ────────────────────────────────────────────────────
def cmd_score():
    print("── Step 3: Score Layouts ───────────────────────────────")
    print("  Evolutionary tournament — 3 rounds of elimination.")
    print("  Press Ctrl+C to stop. Re-run to continue from current round.\n")
    from score import score
    score()


# ── Visualize ────────────────────────────────────────────────
def cmd_visualize():
    print("── Step 4: Visualize Results ───────────────────────────")
    from visualize import plot_training_curve, show_top_layouts, layout_heatmap
    plot_training_curve()
    show_top_layouts(n=20)
    layout_heatmap()
    print("\n✓ Images saved to logs/ and top_layouts/ folders")


# ── All ──────────────────────────────────────────────────────
def cmd_all():
    banner()
    import shutil
    free_gb = shutil.disk_usage(".").free / 1e9

    print("Running full pipeline...\n")
    print(f"Disk free: {free_gb:.1f} GB")

    if free_gb >= 220:
        print("Step 1: Generating all layouts")
        cmd_generate()
    else:
        print("Step 1: Skipping layout generation (not enough disk — will sample on-the-fly)")

    print("\nStep 2: Training")
    cmd_train()

    print("\nStep 3: Scoring")
    cmd_score()

    print("\nStep 4: Visualizing")
    cmd_visualize()

    print("\n══ ALL DONE ══")
    print(f"Top 10,000 layouts: top_layouts/top_10000.npy")
    print(f"Best layout visualization: top_layouts/top_layouts.png")


# ── Help ─────────────────────────────────────────────────────
def cmd_help():
    print("""
Commands:
  python run.py setup       Install deps, check GPU + disk
  python run.py generate    Generate all 15B layouts to disk (needs ~220GB free)
  python run.py train       Train the AI (Ctrl+C saves, re-run resumes)
  python run.py score       Run evolutionary tournament to find top 10K
  python run.py visualize   Generate plots and images
  python run.py status      Show current progress
  python run.py all         Run everything automatically

Typical workflow on your laptop (GTX 1650, 8GB RAM):
  1. python run.py setup
  2. python run.py train        ← runs for ~2-3 days, stop/start anytime
  3. python run.py score        ← runs for ~17 hours
  4. python run.py visualize    ← instant

Output files:
  checkpoints/model_latest.pt       ← trained model
  top_layouts/top_10000.npy         ← best 10,000 layouts (shape: 10000,10,10)
  top_layouts/top_layouts.png       ← visual grid of top 20
  top_layouts/heatmap.png           ← cell occupation heatmap
  logs/training_curve.png           ← loss + turns over time
""")


# ── Entry point ──────────────────────────────────────────────
COMMANDS = {
    "setup":      cmd_setup,
    "generate":   cmd_generate,
    "train":      cmd_train,
    "score":      cmd_score,
    "visualize":  cmd_visualize,
    "status":     cmd_status,
    "all":        cmd_all,
    "help":       cmd_help,
    "--help":     cmd_help,
    "-h":         cmd_help,
}

if __name__ == "__main__":
    banner()

    if len(sys.argv) < 2:
        cmd_help()
        sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        cmd_help()
        sys.exit(1)

    COMMANDS[cmd]()
