# ============================================================
#  validate.py
#  Phase 3: Dual Validation
#
#  Takes the top layouts produced by the GA and re-scores them
#  against BOTH attackers with many more games for accuracy.
#
#  Final score = 0.65 × PDF_score + 0.35 × RL_score
#  (PDF weighted higher as it's the more reliable benchmark)
#
#  RESUME: saves progress per-batch so can be interrupted.
# ============================================================

import os
import json
import time
import signal
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

import torch

import config
from bitboard import grid_to_bits, layout_to_grid, BitGame, BOARD, TOTAL
from pdf_bot import score_layout_worker
from model import load_model

_stop_requested = False

def _handle_signal(sig, frame):
    global _stop_requested
    print("\n[Validate] Stopping after current batch...")
    _stop_requested = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

VAL_PROGRESS_FILE = os.path.join(config.CKPT_DIR, "val_progress.json")


# ── Reconstruct placements from grid (needed for PDF bot) ───
def grid_to_placements(grid):
    """
    Reconstruct ship placements from a (10,10) binary grid.
    Returns list of (row, col, length, horizontal).
    Uses connected-component labeling.
    """
    visited = np.zeros((BOARD, BOARD), dtype=bool)
    placements = []

    for r in range(BOARD):
        for c in range(BOARD):
            if grid[r, c] == 1 and not visited[r, c]:
                # BFS to find connected ship
                cells = []
                q = [(r, c)]
                visited[r, c] = True
                while q:
                    cr, cc = q.pop()
                    cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<BOARD and 0<=nc<BOARD and grid[nr,nc]==1 and not visited[nr,nc]:
                            visited[nr,nc] = True
                            q.append((nr,nc))

                length = len(cells)
                rows   = [x for x,_ in cells]
                cols   = [y for _,y in cells]
                horizontal = (min(rows) == max(rows))
                start_r    = min(rows)
                start_c    = min(cols)
                placements.append((start_r, start_c, length, horizontal))

    return placements


# ── PDF validation (CPU, parallel) ──────────────────────────
def validate_pdf_batch(layouts, n_games, workers):
    """
    layouts: list of (10,10) float32 grids
    Returns: list of float scores
    """
    from bitboard import grid_to_bits
    args = []
    for grid in layouts:
        placements = grid_to_placements(grid)
        bits = grid_to_bits(grid)
        args.append((bits, placements, n_games))

    if workers > 1:
        with Pool(workers) as pool:
            scores = list(tqdm(
                pool.imap(score_layout_worker, args, chunksize=20),
                total=len(args), desc="PDF validation", unit="layout"
            ))
    else:
        scores = [score_layout_worker(a) for a in tqdm(args, desc="PDF validation")]

    return np.array(scores, dtype=np.float32)


# ── RL validation (GPU) ──────────────────────────────────────
def model_attack_layout(model, bits, placements, device, n_games):
    """Play n_games of model attacking a specific layout."""
    total_turns = 0.0

    for _ in range(n_games):
        game = BitGame(bits, placements)
        while not game.done and game.turns < TOTAL:
            state = game.board_state()
            valid = game.valid_shots()
            mask  = np.zeros(TOTAL, dtype=bool)
            for r, c in valid:
                mask[r * BOARD + c] = True

            with torch.no_grad():
                t     = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                probs = model(t).squeeze(0).cpu().numpy()

            probs[~mask] = 0
            s = probs.sum()
            if s == 0:
                probs = mask.astype(np.float32)
                probs /= probs.sum()
            else:
                probs /= s

            action = int(np.argmax(probs))
            r, c   = action // BOARD, action % BOARD
            game.shoot(r, c)

        total_turns += game.turns

    return total_turns / n_games


def validate_rl_batch(layouts, model, device, n_games):
    """
    layouts: list of (10,10) float32 grids
    Returns: np.array of scores
    """
    from bitboard import grid_to_bits
    scores = []

    for grid in tqdm(layouts, desc="RL validation", unit="layout"):
        placements = grid_to_placements(grid)
        bits = grid_to_bits(grid)
        s = model_attack_layout(model, bits, placements, device, n_games)
        scores.append(s)

    return np.array(scores, dtype=np.float32)


# ── Progress save/load ───────────────────────────────────────
def save_val_progress(pdf_scores, rl_scores, n_processed):
    state = {
        'n_processed': n_processed,
        'pdf_scores':  pdf_scores.tolist() if pdf_scores is not None else [],
        'rl_scores':   rl_scores.tolist()  if rl_scores  is not None else [],
    }
    tmp = VAL_PROGRESS_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(state, f)
    os.replace(tmp, VAL_PROGRESS_FILE)


def load_val_progress():
    if not os.path.exists(VAL_PROGRESS_FILE):
        return None, None, 0
    with open(VAL_PROGRESS_FILE) as f:
        state = json.load(f)
    pdf = np.array(state['pdf_scores'], dtype=np.float32) if state['pdf_scores'] else None
    rl  = np.array(state['rl_scores'],  dtype=np.float32) if state['rl_scores']  else None
    return pdf, rl, state['n_processed']


# ── Main validation ──────────────────────────────────────────
def validate():
    global _stop_requested

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    os.makedirs(config.TOP_DIR,  exist_ok=True)

    # Load GA results
    ga_layouts_path = os.path.join(config.CKPT_DIR, "ga_top_layouts.npy")
    ga_scores_path  = os.path.join(config.CKPT_DIR, "ga_top_scores.npy")

    if not os.path.exists(ga_layouts_path):
        raise FileNotFoundError(
            "GA results not found. Run: python run.py ga\n"
            f"Expected: {ga_layouts_path}"
        )

    all_layouts   = np.load(ga_layouts_path)    # (N, 10, 10)
    ga_scores_raw = np.load(ga_scores_path)      # (N,)
    N = len(all_layouts)

    # Only validate top subset if GA produced more than needed
    TOP_N = min(N, config.GA_POPULATION)
    layouts = all_layouts[:TOP_N]
    print(f"\n[Validate] ═══ Dual Validation ═══")
    print(f"  Layouts to validate : {TOP_N:,}")
    print(f"  PDF games each      : {config.VAL_PDF_GAMES}")
    print(f"  RL games each       : {config.VAL_RL_GAMES}  (skipped if no model)")
    print(f"  PDF weight          : {config.VAL_PDF_WEIGHT}")
    print(f"  RL weight           : {config.VAL_RL_WEIGHT}\n")

    # Resume support
    pdf_scores, rl_scores, n_done = load_val_progress()
    if n_done > 0:
        print(f"[Validate] Resuming from {n_done:,} / {TOP_N:,} layouts")

    # ── PDF Validation ───────────────────────────────────────
    print("[Validate] Phase A: PDF bot scoring...")
    if pdf_scores is None or len(pdf_scores) < TOP_N:
        start = len(pdf_scores) if pdf_scores is not None else 0
        remaining = list(layouts[start:])

        new_pdf = validate_pdf_batch(remaining, config.VAL_PDF_GAMES, config.VAL_WORKERS)

        pdf_scores = np.concatenate([pdf_scores, new_pdf]) if pdf_scores is not None else new_pdf
        save_val_progress(pdf_scores, rl_scores, len(pdf_scores))
        print(f"[Validate] PDF scoring complete. "
              f"Mean={pdf_scores.mean():.2f}, Best={pdf_scores.max():.2f}")
    else:
        print(f"[Validate] PDF scores already complete. Best={pdf_scores.max():.2f}")

    if _stop_requested:
        print("[Validate] Stopped after PDF phase. Re-run to continue with RL.")
        return

    # ── RL Validation (if model available) ──────────────────
    rl_available = os.path.exists(config.RL_MODEL_FILE)

    if rl_available:
        print("\n[Validate] Phase B: RL model scoring...")
        device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        model  = load_model(config.RL_MODEL_FILE, device)

        if rl_scores is None or len(rl_scores) < TOP_N:
            start = len(rl_scores) if rl_scores is not None else 0
            remaining = list(layouts[start:])

            new_rl = validate_rl_batch(remaining, model, device, config.VAL_RL_GAMES)
            rl_scores = np.concatenate([rl_scores, new_rl]) if rl_scores is not None else new_rl
            save_val_progress(pdf_scores, rl_scores, TOP_N)
            print(f"[Validate] RL scoring complete. "
                  f"Mean={rl_scores.mean():.2f}, Best={rl_scores.max():.2f}")
        else:
            print(f"[Validate] RL scores already complete. Best={rl_scores.max():.2f}")
    else:
        print("\n[Validate] No RL model found — using PDF scores only.")
        print("  (Train first with: python run.py train_rl)")
        rl_scores = pdf_scores.copy()

    if _stop_requested:
        print("[Validate] Stopped. Re-run to finish.")
        return

    # ── Combined Ranking ─────────────────────────────────────
    print("\n[Validate] Computing combined scores...")

    # Normalize both to [0, 1] range before combining
    pdf_norm = (pdf_scores - pdf_scores.min()) / (pdf_scores.max() - pdf_scores.min() + 1e-8)
    rl_norm  = (rl_scores  - rl_scores.min())  / (rl_scores.max()  - rl_scores.min()  + 1e-8)

    combined = (config.VAL_PDF_WEIGHT * pdf_norm +
                config.VAL_RL_WEIGHT  * rl_norm)

    # Keep top 10,000
    final_n   = min(10_000, TOP_N)
    top_idxs  = np.argsort(combined)[::-1][:final_n]

    top_layouts  = layouts[top_idxs]
    top_combined = combined[top_idxs]
    top_pdf      = pdf_scores[top_idxs]
    top_rl       = rl_scores[top_idxs]

    # ── Save final outputs ───────────────────────────────────
    np.save(config.TOP_LAYOUTS_FILE, top_layouts)
    np.save(config.TOP_SCORES_FILE,  top_combined)

    meta = {
        'n_layouts':       final_n,
        'pdf_weight':      config.VAL_PDF_WEIGHT,
        'rl_weight':       config.VAL_RL_WEIGHT,
        'rl_model_used':   rl_available,
        'pdf_games':       config.VAL_PDF_GAMES,
        'rl_games':        config.VAL_RL_GAMES if rl_available else 0,
        'best_pdf_score':  float(top_pdf[0]),
        'best_rl_score':   float(top_rl[0]),
        'best_combined':   float(top_combined[0]),
        'mean_pdf_top100': float(top_pdf[:100].mean()),
        'mean_pdf_all':    float(top_pdf.mean()),
    }
    with open(config.TOP_META_FILE, 'w') as f:
        json.dump(meta, f, indent=2)

    # Clean up progress file
    if os.path.exists(VAL_PROGRESS_FILE):
        os.remove(VAL_PROGRESS_FILE)

    print(f"\n[Validate] ═══ DONE ═══")
    print(f"  Top {final_n:,} layouts saved to: {config.TOP_LAYOUTS_FILE}")
    print(f"  Best PDF score : {top_pdf[0]:.2f} shots survived")
    if rl_available:
        print(f"  Best RL score  : {top_rl[0]:.2f} shots survived")
    print(f"  Best combined  : {top_combined[0]:.4f} (normalized)")


if __name__ == "__main__":
    validate()
