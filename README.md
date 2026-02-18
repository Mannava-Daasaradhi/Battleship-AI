# Battleship AI — GTX 1650 Edition

Find the top 10,000 most defensible Battleship layouts using self-play
reinforcement learning. Fully stop/resume safe — Ctrl+C anytime.

---

## Quick Start

```bash
# 1. Install dependencies
python run.py setup

# 2. Train the AI  (2–3 days, fully resumable)
python run.py train

# 3. Find top 10,000 layouts  (~17 hours, resumable per-round)
python run.py score

# 4. View results
python run.py visualize
```

Or run everything automatically:
```bash
python run.py all
```

---

## Stop & Resume

**Training:**  
Press `Ctrl+C` at any time. Progress is saved every 50,000 episodes.  
Re-run `python run.py train` — it picks up exactly where it stopped.

**Scoring:**  
Press `Ctrl+C` at any time. Each completed round is saved.  
Re-run `python run.py score` — it skips completed rounds.

Check current progress anytime:
```bash
python run.py status
```

---

## Project Structure

```
battleship_ai/
├── run.py               ← START HERE — master entry point
├── config.py            ← all settings (edit to tune)
├── model.py             ← CNN neural network
├── game_engine.py       ← fast game simulation
├── layout_generator.py  ← generates valid layouts
├── train.py             ← self-play RL training
├── score.py             ← evolutionary tournament
├── visualize.py         ← plots and heatmaps
├── requirements.txt
│
├── data/
│   └── layouts.bin      ← all 15B layouts (~195GB, optional)
│
├── checkpoints/
│   ├── model_latest.pt  ← latest trained model
│   ├── train_state.json ← training progress
│   └── score_state.json ← scoring progress
│
├── logs/
│   ├── train_log.csv    ← episode-by-episode stats
│   └── training_curve.png
│
└── top_layouts/
    ├── top_10000.npy        ← FINAL OUTPUT (shape: 10000,10,10)
    ├── top_10000_scores.npy ← scores for each layout
    ├── top_layouts.png      ← visual grid of top 20
    └── heatmap.png          ← cell occupation heatmap
```

---

## Hardware Requirements

| Component | Minimum | Your Setup |
|-----------|---------|------------|
| GPU | Any NVIDIA | GTX 1650 (4GB) ✓ |
| RAM | 6GB | 8GB ✓ |
| Storage | 50GB | 512GB ✓ |

**Note:** Layout generation (`python run.py generate`) requires 220GB free.  
If you don't have space, skip it — scoring samples layouts on-the-fly.

---

## Estimated Timelines (GTX 1650)

| Step | Time |
|------|------|
| Setup | 2 minutes |
| Generate all layouts (optional) | 4–6 hours (CPU) |
| Train 5M episodes | 2–3 days |
| Score (3 rounds) | ~17 hours |
| Visualize | 1 minute |

---

## Using the Results

```python
import numpy as np

# Load top 10,000 layouts
layouts = np.load("top_layouts/top_10000.npy")   # shape: (10000, 10, 10)
scores  = np.load("top_layouts/top_10000_scores.npy")

# The #1 best layout
best_layout = layouts[0]   # (10, 10) array, 1=ship, 0=water
print(f"Best layout survives avg {scores[0]:.1f} turns against trained AI")

# Display it
for row in best_layout:
    print("".join("█" if c else "·" for c in row))
```

---

## Tuning config.py

```python
# Make training faster (less accurate)
TOTAL_EPISODES = 1_000_000   # default: 5M
BATCH_SIZE = 128             # default: 256

# Make scoring faster (less accurate top-10K)
ROUND1_SAMPLE = 1_000_000    # default: 5M
ROUND1_GAMES  = 10           # default: 20

# If VRAM errors occur
BATCH_SIZE = 64
SCORE_BATCH_SIZE = 512
MIXED_PRECISION = False
```
