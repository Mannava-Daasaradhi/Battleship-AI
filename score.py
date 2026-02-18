# ============================================================
#  score.py
#  Evolutionary tournament to find the top 10,000 layouts.
#
#  3-round elimination:
#    Round 1: Sample 5M layouts, 20 games each → keep top 500K
#    Round 2: 500K layouts, 100 games each     → keep top 50K
#    Round 3: 50K layouts,  500 games each     → keep top 10K
#
#  RESUME: Each round saves progress. If interrupted, restarts
#  from the beginning of the current round (fast enough).
# ============================================================

import os
import json
import time
import signal
import numpy as np
from tqdm import tqdm

import torch

import config
from model import build_model
from layout_generator import load_layouts_chunk, count_layouts, random_layout, LAYOUTS_FILE
from game_engine import BattleshipGame

_stop_requested = False

def _handle_signal(sig, frame):
    global _stop_requested
    print("\n[Score] Stop requested — finishing current batch and saving...")
    _stop_requested = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── Load trained model ───────────────────────────────────────
def load_model(device):
    if not os.path.exists(config.MODEL_FILE):
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_FILE}\n"
            "Run training first: python run.py train"
        )
    model = build_model(device)
    ckpt  = torch.load(config.MODEL_FILE, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"[Score] Loaded model from {config.MODEL_FILE}")
    return model


# ── Score a batch of layouts (N games each) ─────────────────
def score_batch(layouts: np.ndarray, model, device, n_games: int):
    """
    layouts : (N, 10, 10) float32
    Returns : (N,) float32 — average turns survived (higher = better)
    """
    N = len(layouts)
    total_turns = np.zeros(N, dtype=np.float32)

    for _ in range(n_games):
        for i in range(N):
            game = BattleshipGame(layouts[i])
            while not game.done and game.turns < 100:
                state = game.state()
                mask  = game.valid_shot_mask()
                with torch.no_grad():
                    t     = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    probs = model(t).squeeze(0).cpu().numpy()
                probs[~mask] = 0
                if probs.sum() == 0:
                    probs = mask.astype(np.float32)
                probs /= probs.sum()
                action = int(np.random.choice(config.BOARD_SIZE**2, p=probs))
                r, c   = divmod(action, config.BOARD_SIZE)
                game.shoot(r, c)
            total_turns[i] += game.turns

    return total_turns / n_games


# ── Sample random layouts from file OR generate on the fly ───
def sample_random_layouts(n: int, use_file: bool = True):
    """Returns (n, 10, 10) array of random valid layouts."""
    if use_file and os.path.exists(config.LAYOUTS_FILE):
        total = count_layouts(config.LAYOUTS_FILE)
        idxs  = np.random.choice(total, size=n, replace=False)
        idxs.sort()

        layouts = []
        batch_size = 100_000
        ptr = 0
        while ptr < len(idxs):
            batch_idxs = idxs[ptr:ptr+batch_size]
            # Load each by index (sequential read optimization)
            for idx in batch_idxs:
                chunk = load_layouts_chunk(config.LAYOUTS_FILE, int(idx), 1)
                if chunk is not None:
                    layouts.append(chunk[0])
            ptr += batch_size
        return np.array(layouts)
    else:
        # Generate randomly without the file
        print("[Score] layouts.bin not found — generating layouts on the fly")
        return np.array([random_layout() for _ in range(n)])


# ── Single round of scoring ──────────────────────────────────
def run_round(layouts, model, device, n_games, keep_n, round_name):
    """
    Score all layouts in batches, return top keep_n.
    """
    N          = len(layouts)
    scores     = np.zeros(N, dtype=np.float32)
    batch_size = config.SCORE_BATCH_SIZE

    print(f"\n[Score] ── {round_name} ──")
    print(f"  Layouts: {N:,}  |  Games each: {n_games}  |  Keeping: {keep_n:,}")

    t0 = time.time()
    for start in tqdm(range(0, N, batch_size), desc=round_name, unit="batch"):
        if _stop_requested:
            print("[Score] Interrupted — returning partial results")
            break
        end   = min(start + batch_size, N)
        batch = layouts[start:end]
        scores[start:end] = score_batch(batch, model, device, n_games)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min")

    top_idxs  = np.argsort(scores)[-keep_n:][::-1]
    top_layouts = layouts[top_idxs]
    top_scores  = scores[top_idxs]

    print(f"  Score range: {top_scores[-1]:.1f} – {top_scores[0]:.1f} turns")
    return top_layouts, top_scores


# ── Load/save intermediate results ───────────────────────────
def save_round_result(round_num, layouts, scores):
    path_l = os.path.join(config.CKPT_DIR, f"round{round_num}_layouts.npy")
    path_s = os.path.join(config.CKPT_DIR, f"round{round_num}_scores.npy")
    np.save(path_l, layouts)
    np.save(path_s, scores)

    state = {'round_complete': round_num}
    with open(config.SCORE_CKPT_FILE, 'w') as f:
        json.dump(state, f)
    print(f"[Score] ✓ Round {round_num} saved")


def load_round_result(round_num):
    path_l = os.path.join(config.CKPT_DIR, f"round{round_num}_layouts.npy")
    path_s = os.path.join(config.CKPT_DIR, f"round{round_num}_scores.npy")
    if os.path.exists(path_l) and os.path.exists(path_s):
        return np.load(path_l), np.load(path_s)
    return None, None


def get_completed_rounds():
    if not os.path.exists(config.SCORE_CKPT_FILE):
        return 0
    with open(config.SCORE_CKPT_FILE) as f:
        state = json.load(f)
    return state.get('round_complete', 0)


# ── Main scoring pipeline ────────────────────────────────────
def score():
    global _stop_requested

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    os.makedirs(config.TOP_DIR,  exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[Score] Device: {device}")

    model = load_model(device)

    completed = get_completed_rounds()
    print(f"[Score] Completed rounds: {completed}")

    # ── Round 1 ─────────────────────────────────────────────
    if completed >= 1:
        print("[Score] Round 1 already complete — loading saved results")
        r1_layouts, r1_scores = load_round_result(1)
    else:
        print(f"\n[Score] Sampling {config.ROUND1_SAMPLE:,} layouts for Round 1...")
        use_file = os.path.exists(config.LAYOUTS_FILE)

        if use_file:
            # Efficient chunked sampling from file
            total   = count_layouts(config.LAYOUTS_FILE)
            sample_n = min(config.ROUND1_SAMPLE, total)
            print(f"  Layout file: {total/1e9:.2f}B layouts available, sampling {sample_n/1e6:.0f}M")

            # Sample in chunks to avoid RAM explosion
            chunk_size    = 50_000
            all_layouts   = []
            sampled_idxs  = np.random.choice(total, size=sample_n, replace=False)
            sampled_idxs.sort()

            print("  Loading sampled layouts from disk...")
            for i in tqdm(range(0, len(sampled_idxs), chunk_size)):
                batch_idxs = sampled_idxs[i:i+chunk_size]
                for idx in batch_idxs:
                    chunk = load_layouts_chunk(config.LAYOUTS_FILE, int(idx), 1)
                    if chunk is not None:
                        all_layouts.append(chunk[0])

            r1_pool = np.array(all_layouts, dtype=np.float32)
        else:
            print("  Generating layouts on-the-fly (layouts.bin not found)")
            r1_pool = np.array([random_layout() for _ in tqdm(range(config.ROUND1_SAMPLE))])

        r1_layouts, r1_scores = run_round(
            r1_pool, model, device,
            n_games=config.ROUND1_GAMES,
            keep_n=config.ROUND1_KEEP,
            round_name="Round 1"
        )
        if not _stop_requested:
            save_round_result(1, r1_layouts, r1_scores)

    if _stop_requested:
        print("[Score] Stopped after Round 1. Re-run to continue from Round 2.")
        return

    # ── Round 2 ─────────────────────────────────────────────
    if completed >= 2:
        print("[Score] Round 2 already complete — loading saved results")
        r2_layouts, r2_scores = load_round_result(2)
    else:
        r2_layouts, r2_scores = run_round(
            r1_layouts, model, device,
            n_games=config.ROUND2_GAMES,
            keep_n=config.ROUND2_KEEP,
            round_name="Round 2"
        )
        if not _stop_requested:
            save_round_result(2, r2_layouts, r2_scores)

    if _stop_requested:
        print("[Score] Stopped after Round 2. Re-run to continue from Round 3.")
        return

    # ── Round 3 ─────────────────────────────────────────────
    if completed >= 3:
        print("[Score] Round 3 already complete — loading saved results")
        r3_layouts, r3_scores = load_round_result(3)
    else:
        r3_layouts, r3_scores = run_round(
            r2_layouts, model, device,
            n_games=config.ROUND3_GAMES,
            keep_n=config.ROUND3_KEEP,
            round_name="Round 3"
        )
        if not _stop_requested:
            save_round_result(3, r3_layouts, r3_scores)

    if _stop_requested:
        print("[Score] Stopped after Round 3.")
        return

    # ── Save final top 10K ───────────────────────────────────
    os.makedirs(config.TOP_DIR, exist_ok=True)
    np.save(config.TOP_LAYOUTS_FILE, r3_layouts)
    np.save(os.path.join(config.TOP_DIR, "top_10000_scores.npy"), r3_scores)

    print(f"\n[Score] ═══ DONE ═══")
    print(f"  Top 10,000 layouts saved to: {config.TOP_LAYOUTS_FILE}")
    print(f"  Best layout score : {r3_scores[0]:.2f} turns survived (avg)")
    print(f"  Worst of top 10K  : {r3_scores[-1]:.2f} turns survived (avg)")

    # Print the single best layout
    print("\n  Best layout (1=ship, .=water):")
    best = r3_layouts[0]
    for row in range(config.BOARD_SIZE):
        line = ""
        for col in range(config.BOARD_SIZE):
            line += "█" if best[row, col] == 1 else "·"
        print(f"    {line}")


if __name__ == "__main__":
    score()
