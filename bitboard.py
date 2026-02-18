# ============================================================
#  bitboard.py
#  Ultra-fast Battleship board operations using Python integers
#  as 100-bit bitboards.  No NumPy in the hot path.
#
#  Bit index = row * 10 + col  (bit 0 = top-left, bit 99 = bottom-right)
#
#  Layout representation:
#    Ship = list of (row, col, length, horizontal) tuples
#    Bitboard = Python int with 1-bits where ships are
# ============================================================

import numpy as np
import config

BOARD  = config.BOARD_SIZE
SHIPS  = config.SHIPS
TOTAL  = BOARD * BOARD   # 100 cells


# ── Bit helpers ──────────────────────────────────────────────
def cell_bit(row, col):
    return 1 << (row * BOARD + col)


def ship_bits(row, col, length, horizontal):
    """
    Return bitmask for a ship placement, or 0 if out of bounds.
    """
    bits = 0
    for i in range(length):
        r = row if horizontal else row + i
        c = col + i if horizontal else col
        if r >= BOARD or c >= BOARD:
            return 0
        bits |= cell_bit(r, c)
    return bits


def bits_to_grid(bits):
    """Convert 100-bit int → (10,10) float32 numpy array."""
    grid = np.zeros(TOTAL, dtype=np.float32)
    for i in range(TOTAL):
        if bits & (1 << i):
            grid[i] = 1.0
    return grid.reshape(BOARD, BOARD)


def grid_to_bits(grid):
    """Convert (10,10) numpy array → 100-bit int."""
    flat = grid.flatten().astype(int)
    bits = 0
    for i, v in enumerate(flat):
        if v:
            bits |= (1 << i)
    return bits


# ── Ship encoding ─────────────────────────────────────────────
# A layout is a list of ShipPlacement namedtuple-like tuples:
#   (row, col, length, horizontal)
# Plus its combined bitmask for fast overlap checking.

def encode_layout(ship_placements):
    """
    ship_placements: list of (row, col, length, horizontal)
    Returns (bits, placements) or None if any ship is invalid/overlapping.
    """
    combined = 0
    for row, col, length, horizontal in ship_placements:
        b = ship_bits(row, col, length, horizontal)
        if b == 0:
            return None
        if combined & b:
            return None  # overlap
        combined |= b
    return combined, ship_placements


def random_layout():
    """
    Generate one random valid layout.
    Returns (bits, placements) where placements is a list of
    (row, col, length, horizontal) tuples.
    """
    for _attempt in range(10_000):
        combined  = 0
        placements = []
        ok = True

        for length in SHIPS:
            placed = False
            for _inner in range(200):
                row   = np.random.randint(0, BOARD)
                col   = np.random.randint(0, BOARD)
                horiz = bool(np.random.randint(0, 2))
                b     = ship_bits(row, col, length, horiz)
                if b and not (combined & b):
                    combined  |= b
                    placements.append((row, col, length, horiz))
                    placed = True
                    break
            if not placed:
                ok = False
                break

        if ok:
            return combined, placements

    raise RuntimeError("Failed to generate random layout after 10K attempts")


def layout_to_grid(bits):
    """100-bit int → (10,10) float32 array."""
    return bits_to_grid(bits)


# ── Fast game simulation (bitboard-based) ───────────────────
class BitGame:
    """
    Simulates one Battleship game using bitboard operations.
    ~10x faster than NumPy-based BattleshipGame for single-game simulation.
    """

    __slots__ = ['ship_bits_list', 'alive_bits', 'full_bits',
                 'hit_bits', 'miss_bits', 'shots_bits',
                 'turns', 'done', 'total_cells']

    def __init__(self, layout_bits, ship_placements):
        self.ship_bits_list = []
        for row, col, length, horiz in ship_placements:
            self.ship_bits_list.append(ship_bits(row, col, length, horiz))

        self.alive_bits  = layout_bits       # remaining un-sunk ship cells
        self.full_bits   = layout_bits       # all ship cells (constant)
        self.hit_bits    = 0
        self.miss_bits   = 0
        self.shots_bits  = 0
        self.turns       = 0
        self.done        = False
        self.total_cells = bin(layout_bits).count('1')

    def shoot(self, row, col):
        """Returns (hit, sunk_ship_len, done)."""
        b = cell_bit(row, col)
        assert not (self.shots_bits & b), "Duplicate shot"
        self.shots_bits |= b
        self.turns      += 1

        if self.full_bits & b:
            self.hit_bits  |= b
            self.alive_bits &= ~b

            # Check if a ship sunk
            sunk_len = 0
            for sb in self.ship_bits_list:
                if (sb & b) and not (self.alive_bits & sb):
                    sunk_len = bin(sb).count('1')
                    break

            if not self.alive_bits:
                self.done = True

            return True, sunk_len, self.done
        else:
            self.miss_bits |= b
            return False, 0, False

    def unknown_bits(self):
        """Bitmask of cells not yet shot."""
        all_shot = self.shots_bits
        unknown  = 0
        for i in range(TOTAL):
            if not (all_shot & (1 << i)):
                unknown |= (1 << i)
        return unknown

    def board_state(self):
        """
        Returns (3, 10, 10) float32 for neural network:
          ch0: unknown cells
          ch1: hits
          ch2: misses
        """
        unknown = np.zeros(TOTAL, dtype=np.float32)
        hits    = np.zeros(TOTAL, dtype=np.float32)
        misses  = np.zeros(TOTAL, dtype=np.float32)
        for i in range(TOTAL):
            b = 1 << i
            if self.shots_bits & b:
                if self.hit_bits & b:
                    hits[i] = 1.0
                else:
                    misses[i] = 1.0
            else:
                unknown[i] = 1.0
        return np.stack([
            unknown.reshape(BOARD, BOARD),
            hits.reshape(BOARD, BOARD),
            misses.reshape(BOARD, BOARD),
        ], axis=0)

    def valid_shots(self):
        """List of (row, col) not yet shot."""
        shots = []
        for i in range(TOTAL):
            if not (self.shots_bits & (1 << i)):
                shots.append(divmod(i, BOARD))
        return shots
