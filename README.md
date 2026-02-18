# Hybrid Battleship AI — GTX 1650 Edition

Combines the best of two approaches to find the top 10,000 hardest-to-sink
Battleship layouts:

| | Their Approach | Our Approach | **This Hybrid** |
|---|---|---|---|
| Attacker | PDF Bot (optimal) | RL model | **Both** |
| Optimizer | Genetic Algorithm | Tournament filter | **GA (breeds, not just filters)** |
| Validation | Single judge | Single judge | **Dual judge (PDF + RL combined)** |
| Time | ~12-24h | ~3-4 days | **~20-48h** |

---

## Quick Start

```bash
python run.py setup       # check hardware, install deps
python run.py ga          # Phase 1: breed layouts (~16h, Ctrl+C resumable)
python run.py validate    # Phase 3: final scoring (~3h)
python run.py visualize   # generate plots
```

For best results, also run RL training between GA and validate:
```bash
python run.py train_rl    # Phase 2: optional, improves validation (~12-24h)
```

Check progress anytime: `python run.py status`

---

## How Each Phase Works

### Phase 1: Genetic Algorithm + PDF Bot (CPU)

The PDF bot is the mathematically near-optimal Battleship attacker.
After every shot it calculates exactly which cells are most likely to
contain a ship given all previous hits/misses — no training needed.

The GA uses it as a fitness judge:

```
Generation 0:  10,000 random valid layouts
   ↓  Score each against PDF bot (100 games)
   ↓  Keep top 50% as parents
   ↓  Breed: take ships from Parent A + Parent B (crossover)
   ↓  Mutate: randomly move one ship (15% chance)
Generation 1:  new 10,000 layouts (smarter than gen 0)
   ↓  ...
Generation 500: converged near-optimal population
```

Key insight: GA **breeds** good layouts together, not just filters them.
If layout A has great ship dispersion and layout B avoids the center,
their child inherits both traits.

### Phase 2: RL Training (GPU, optional)

A CNN learns to attack by playing thousands of games.
- First 200K episodes: warms up against random layouts with the PDF bot
  strategy as teacher signal
- Remaining episodes: pure self-play for discovering non-obvious patterns

The RL model becomes a **second independent attacker** with different
blind spots than the PDF bot — catching layouts that fooled the PDF bot.

### Phase 3: Dual Validation (CPU + GPU)

Re-scores all GA survivors against both attackers with many more games:
- PDF bot: 500 games per layout (rigorous statistical estimate)
- RL model: 200 games per layout (if model trained)

Final score = 0.65 × PDF_score + 0.35 × RL_score

Layouts that fool both attackers rank highest.

---

## Stop & Resume

Every phase saves its state automatically:

| Phase | Saves every | Resume behavior |
|---|---|---|
| GA | 10 generations | Resumes from exact generation |
| RL Training | 50K episodes | Resumes from exact episode |
| Validation | Each batch | Resumes from last saved batch |

Just press `Ctrl+C` and re-run the same command.

---

## File Structure

```
battleship_hybrid/
├── run.py                ← START HERE
├── config.py             ← all settings (tune here)
├── bitboard.py           ← fast integer-based board operations
├── pdf_bot.py            ← probability density attacker
├── genetic_algorithm.py  ← GA optimizer
├── model.py              ← RL CNN model
├── train_rl.py           ← RL self-play training
├── validate.py           ← dual scoring phase
├── visualize.py          ← plots and heatmaps
│
├── checkpoints/
│   ├── ga_state.json         ← GA progress
│   ├── ga_top_layouts.npy    ← GA output layouts
│   ├── rl_model_latest.pt    ← RL model weights
│   └── rl_train_state.json   ← RL training progress
│
├── logs/
│   ├── ga_log.csv
│   ├── ga_curve.png
│   ├── rl_log.csv
│   └── rl_curve.png
│
└── top_layouts/
    ├── top_10000.npy         ← FINAL OUTPUT
    ├── top_10000_scores.npy
    ├── top_10000_meta.json
    ├── top_layouts.png       ← visual grid of top 20
    └── heatmap.png           ← cell occupation heatmap
```

---

## Tuning for Your Hardware

Edit `config.py`:

```python
# Fewer CPU workers if RAM is low
GA_WORKERS = 2        # default 4

# Fewer generations for faster results
GA_GENERATIONS = 100  # default 500

# Reduce if VRAM errors occur
RL_BATCH_SIZE = 128   # default 256

# Skip RL entirely — GA + PDF validation is still excellent
# Just run: ga → validate → visualize
```

---

## Using the Results

```python
import numpy as np

# Load top 10,000 layouts
layouts = np.load("top_layouts/top_10000.npy")   # (10000, 10, 10)
scores  = np.load("top_layouts/top_10000_scores.npy")

# Best layout
best = layouts[0]
print(f"Score: {scores[0]:.4f}")
for row in best:
    print("".join("█" if c else "·" for c in row))
```
