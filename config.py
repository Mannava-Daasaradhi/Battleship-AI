# ============================================================
#  config.py  —  Master config for the Hybrid Battleship AI
#  Edit values here to tune for your hardware / time budget
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "data")
CKPT_DIR         = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR          = os.path.join(BASE_DIR, "logs")
TOP_DIR          = os.path.join(BASE_DIR, "top_layouts")

# GA checkpoints
GA_CKPT_FILE     = os.path.join(CKPT_DIR, "ga_state.json")
GA_POP_FILE      = os.path.join(CKPT_DIR, "ga_population.npy")
GA_SCORES_FILE   = os.path.join(CKPT_DIR, "ga_scores.npy")

# RL checkpoints
RL_MODEL_FILE    = os.path.join(CKPT_DIR, "rl_model_latest.pt")
RL_TRAIN_FILE    = os.path.join(CKPT_DIR, "rl_train_state.json")

# Final outputs
TOP_LAYOUTS_FILE  = os.path.join(TOP_DIR, "top_10000.npy")
TOP_SCORES_FILE   = os.path.join(TOP_DIR, "top_10000_scores.npy")
TOP_META_FILE     = os.path.join(TOP_DIR, "top_10000_meta.json")

# ── Board & Fleet ────────────────────────────────────────────
BOARD_SIZE       = 10
SHIPS            = [5, 4, 3, 3, 2]   # ship lengths (two 3s are identical)

# ── ═══════════════════════════════════════════════════════ ──
#    PHASE 1: GENETIC ALGORITHM  (CPU)
# ── ═══════════════════════════════════════════════════════ ──

GA_POPULATION    = 10_000     # layouts per generation
GA_GENERATIONS   = 500        # number of breeding cycles
GA_FITNESS_GAMES = 100        # PDF-bot games per layout per fitness eval
GA_ELITE_FRAC    = 0.10       # top 10% survive unchanged (elitism)
GA_SELECT_FRAC   = 0.50       # top 50% eligible to be parents
GA_MUTATE_PROB   = 0.15       # chance a child gets one ship moved
GA_CROSSOVER_MAX_RETRIES = 30 # attempts to resolve ship overlaps after crossover
GA_SAVE_EVERY    = 10         # save checkpoint every N generations
GA_WORKERS       = 4          # parallel CPU workers for fitness eval
                               # (set to your core count - 1, max 6 for 8GB RAM)

# ── ═══════════════════════════════════════════════════════ ──
#    PHASE 2: RL TRAINING  (GPU)
# ── ═══════════════════════════════════════════════════════ ──

RL_TOTAL_EPISODES     = 3_000_000   # self-play games
RL_WARMUP_EPISODES    = 200_000     # episodes using PDF bot as opponent first
RL_BATCH_SIZE         = 256
RL_REPLAY_BUFFER_SIZE = 50_000
RL_LEARNING_RATE      = 1e-4
RL_GAMMA              = 0.99
RL_SAVE_EVERY         = 50_000
RL_LOG_EVERY          = 1_000
RL_MIXED_PRECISION    = True

# CNN architecture (small enough for 4GB VRAM GTX 1650)
RL_CONV_CHANNELS      = [32, 64, 64]
RL_FC_DIMS            = [512, 256]
RL_INPUT_CHANNELS     = 3          # unknown / hit / miss

# ── ═══════════════════════════════════════════════════════ ──
#    PHASE 3: DUAL VALIDATION  (CPU + GPU)
# ── ═══════════════════════════════════════════════════════ ──

VAL_PDF_GAMES    = 500     # PDF-bot games per layout (rigorous)
VAL_RL_GAMES     = 200     # RL-bot games per layout
VAL_PDF_WEIGHT   = 0.65    # weight of PDF score in final ranking
VAL_RL_WEIGHT    = 0.35    # weight of RL score in final ranking
VAL_BATCH_SIZE   = 256     # layouts validated per GPU batch (RL)
VAL_WORKERS      = 4       # CPU workers for PDF validation

# ── Hardware ─────────────────────────────────────────────────
DEVICE           = "cuda"   # auto-falls back to "cpu" in code

# ── Scoring semantics ────────────────────────────────────────
# "score" always = avg shots the attacker needed to win
# Higher score = layout survived longer = BETTER layout
