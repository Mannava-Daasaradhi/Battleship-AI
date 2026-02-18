# ============================================================
#  pdf_bot.py
#  Probability Density Function attacker — the mathematical
#  near-optimal Battleship AI.
#
#  After every shot it asks: "given all hits and misses so far,
#  where is each remaining ship most likely to be?"
#
#  Algorithm:
#    1. For each unsunk ship, enumerate ALL valid positions
#       consistent with current board state (no misses, all
#       visible hits covered)
#    2. Build heatmap: count how many valid configs include each cell
#    3. Shoot the highest-probability unknown cell
#
#  This is provably near-optimal and requires ZERO training.
# ============================================================

import numpy as np
import config
from bitboard import ship_bits, cell_bit, TOTAL, BOARD

SHIPS = config.SHIPS


# ── Core heatmap computation ─────────────────────────────────
def compute_heatmap(hit_bits, miss_bits, shots_bits,
                    remaining_ship_lens, alive_hit_bits):
    """
    Compute probability heatmap for remaining ships.

    Parameters
    ----------
    hit_bits          : int  — cells that are confirmed hits (not yet sunk)
    miss_bits         : int  — cells that are confirmed misses
    shots_bits        : int  — all cells fired at
    remaining_ship_lens : list[int] — lengths of ships not yet sunk
    alive_hit_bits    : int  — hit cells from ships still alive (must be covered)

    Returns
    -------
    heatmap : (10, 10) float32 ndarray — higher = more likely to contain a ship
    """
    heatmap = np.zeros(TOTAL, dtype=np.float32)

    # Unknown cells = not yet shot
    unknown_bits = (~shots_bits) & ((1 << TOTAL) - 1)

    for length in remaining_ship_lens:
        for row in range(BOARD):
            for col in range(BOARD):
                for horiz in (True, False):
                    b = ship_bits(row, col, length, horiz)
                    if not b:
                        continue

                    # Rule 1: No ship cell can be a miss
                    if b & miss_bits:
                        continue

                    # Rule 2: All cells must be unknown or a hit
                    # (can't place ship where another ship was confirmed)
                    if b & ~(unknown_bits | hit_bits):
                        continue

                    # Valid position — add to heatmap
                    pos = b
                    idx = 0
                    while pos:
                        lsb = pos & (-pos)
                        bit_idx = lsb.bit_length() - 1
                        heatmap[bit_idx] += 1.0
                        pos &= pos - 1

    # Zero out already-shot cells (can't shoot there again)
    for i in range(TOTAL):
        if shots_bits & (1 << i):
            heatmap[i] = 0.0

    return heatmap.reshape(BOARD, BOARD)


# ── PDF Bot player ────────────────────────────────────────────
class PDFBot:
    """
    Stateful PDF bot that plays one full game.
    Usage:
        bot = PDFBot(layout_bits, ship_placements)
        while not bot.game.done:
            bot.step()
        print(bot.game.turns)
    """

    def __init__(self, layout_bits, ship_placements):
        from bitboard import BitGame
        self.game           = BitGame(layout_bits, ship_placements)
        self.ship_placements = ship_placements
        self.ship_bits_list  = []
        from bitboard import ship_bits as sb
        for r, c, l, h in ship_placements:
            self.ship_bits_list.append(sb(r, c, l, h))

        self.sunk_lengths   = []      # lengths of sunk ships
        self.unsunk_lengths = list(SHIPS)  # ship lengths still alive

    def _alive_hit_bits(self):
        """Bits of cells hit that belong to still-alive ships."""
        return self.game.hit_bits

    def step(self):
        """Fire one shot using PDF heuristic."""
        if self.game.done:
            return

        heatmap = compute_heatmap(
            hit_bits           = self.game.hit_bits,
            miss_bits          = self.game.miss_bits,
            shots_bits         = self.game.shots_bits,
            remaining_ship_lens= self.unsunk_lengths,
            alive_hit_bits     = self._alive_hit_bits(),
        )

        # Pick highest-probability unknown cell
        # Add tiny random noise to break ties randomly
        heatmap += np.random.uniform(0, 1e-6, heatmap.shape)
        idx = int(np.argmax(heatmap))
        row, col = divmod(idx, BOARD)

        hit, sunk_len, done = self.game.shoot(row, col)

        if sunk_len > 0:
            self.unsunk_lengths.remove(sunk_len)
            self.sunk_lengths.append(sunk_len)

    def play_full_game(self):
        """Play to completion. Returns total turns taken."""
        while not self.game.done and self.game.turns < TOTAL:
            self.step()
        return self.game.turns


# ── Convenience: score one layout ────────────────────────────
def score_layout_pdf(layout_bits, ship_placements, n_games=100):
    """
    Play n_games against the PDF bot.
    Returns average shots survived (higher = better layout).
    """
    total = 0
    for _ in range(n_games):
        bot = PDFBot(layout_bits, ship_placements)
        total += bot.play_full_game()
    return total / n_games


# ── Batch scoring (for GA fitness, multiprocessing-friendly) ─
def score_layout_worker(args):
    """
    Worker function for multiprocessing.Pool.map().
    args = (layout_bits, ship_placements, n_games)
    Returns float score.
    """
    layout_bits, ship_placements, n_games = args
    return score_layout_pdf(layout_bits, ship_placements, n_games)


# ── Quick self-test ───────────────────────────────────────────
if __name__ == "__main__":
    import time
    from bitboard import random_layout

    print("Testing PDF bot on 10 random layouts (100 games each)...")
    scores = []
    t0 = time.time()
    for i in range(10):
        bits, placements = random_layout()
        s = score_layout_pdf(bits, placements, n_games=100)
        scores.append(s)
        print(f"  Layout {i+1}: {s:.1f} shots survived")

    elapsed = time.time() - t0
    print(f"\nAvg: {np.mean(scores):.1f} shots | Time: {elapsed:.1f}s")
    print(f"Speed: {10*100/elapsed:.0f} games/sec")
