# ============================================================
#  config.py  —  All settings in one place
#  Edit these values to tune for your hardware
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
CKPT_DIR        = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
TOP_DIR         = os.path.join(BASE_DIR, "top_layouts")

LAYOUTS_FILE    = os.path.join(DATA_DIR, "layouts.bin")
SCORES_FILE     = os.path.join(DATA_DIR, "scores.bin")
PROGRESS_FILE   = os.path.join(DATA_DIR, "progress.json")
TOP_LAYOUTS_FILE= os.path.join(TOP_DIR,  "top_10000.npy")
MODEL_FILE      = os.path.join(CKPT_DIR, "model_latest.pt")
TRAIN_CKPT_FILE = os.path.join(CKPT_DIR, "train_state.json")
SCORE_CKPT_FILE = os.path.join(CKPT_DIR, "score_state.json")

# ── Board & Fleet ────────────────────────────────────────────
BOARD_SIZE      = 10
SHIPS           = [5, 4, 3, 3, 2]   # lengths; two 3s are identical
IDENTICAL_SHIPS = [(2, 3)]          # (index_a, index_b) of identical pairs

# ── Training ─────────────────────────────────────────────────
TOTAL_EPISODES      = 5_000_000     # self-play games to train on
BATCH_SIZE          = 256           # keep low for 4GB VRAM
REPLAY_BUFFER_SIZE  = 50_000        # max experiences in RAM
LEARNING_RATE       = 1e-4
GAMMA               = 0.99          # reward discount
SAVE_EVERY          = 50_000        # save checkpoint every N episodes
LOG_EVERY           = 1_000         # print stats every N episodes
MIXED_PRECISION     = True          # ~30% speedup, halves VRAM usage

# ── Scoring (Evolutionary) ───────────────────────────────────
ROUND1_SAMPLE       = 5_000_000     # layouts sampled in round 1
ROUND1_GAMES        = 20            # games per layout in round 1
ROUND1_KEEP         = 500_000       # survivors after round 1

ROUND2_GAMES        = 100           # games per layout in round 2
ROUND2_KEEP         = 50_000        # survivors after round 2

ROUND3_GAMES        = 500           # games per layout in round 3
ROUND3_KEEP         = 10_000        # final top layouts

SCORE_BATCH_SIZE    = 2048          # layouts scored per GPU batch

# ── Model Architecture ───────────────────────────────────────
CONV_CHANNELS       = [32, 64, 64]  # CNN layer channels
FC_DIMS             = [512, 256]    # fully connected dims
INPUT_CHANNELS      = 3            # hit / miss / unknown

# ── Hardware ─────────────────────────────────────────────────
NUM_WORKERS         = 2            # dataloader workers (keep low, 8GB RAM)
PIN_MEMORY          = True
DEVICE              = "cuda"       # falls back to cpu automatically in code

# ── Generation ───────────────────────────────────────────────
GEN_CHUNK_SIZE      = 100_000      # layouts written per disk flush
BYTES_PER_LAYOUT    = 13           # np.packbits(100 bits) = 13 bytes
