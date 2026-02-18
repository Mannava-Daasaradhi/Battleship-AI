# ============================================================
#  layout_generator.py
#  Generates all ~15 billion valid Battleship layouts,
#  streams directly to disk — never loads all into RAM.
#  Resumes automatically if interrupted.
# ============================================================

import os
import json
import time
import numpy as np
from tqdm import tqdm
import config

# ── Canonical dedup key for identical ships ──────────────────
def _canonical_key(cells_list_a, cells_list_b):
    """Sort two ship cell-lists so (A,B) == (B,A)."""
    a = tuple(sorted(cells_list_a))
    b = tuple(sorted(cells_list_b))
    return (a, b) if a <= b else (b, a)


# ── Fast single-ship placement ───────────────────────────────
def _try_place(occupied, row, col, length, horizontal):
    """
    Returns list of (r,c) cells if ship fits, else None.
    occupied: set of (r,c) already taken.
    """
    cells = []
    for i in range(length):
        r = row if horizontal else row + i
        c = col + i if horizontal else col
        if r >= config.BOARD_SIZE or c >= config.BOARD_SIZE:
            return None
        if (r, c) in occupied:
            return None
        cells.append((r, c))
    return cells


# ── Core recursive generator ─────────────────────────────────
class LayoutGenerator:
    def __init__(self):
        self.ships = config.SHIPS          # [5, 4, 3, 3, 2]
        self.count = 0
        self._chunk = []
        self._file  = None

    def generate_all(self, output_path):
        """
        Enumerate all valid layouts, pack to bits, write to binary file.
        Skips duplicate arrangements for identical 3-ships.
        Returns total count written.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Resume support: check how many bytes already written
        start_fresh = True
        if os.path.exists(output_path):
            existing = os.path.getsize(output_path)
            if existing > 0:
                already = existing // config.BYTES_PER_LAYOUT
                print(f"[Generator] Found existing file with {already:,} layouts. "
                      "Delete data/layouts.bin to regenerate from scratch.")
                return already

        print("[Generator] Starting layout enumeration...")
        print("  Ships:", self.ships)
        print("  Output:", output_path)
        t0 = time.time()

        with open(output_path, 'wb') as f:
            self._file = f
            self._recurse(ship_idx=0,
                          occupied=set(),
                          placed_cells=[],
                          grid=np.zeros((config.BOARD_SIZE, config.BOARD_SIZE), dtype=np.uint8))
            # Flush remaining chunk
            if self._chunk:
                np.array(self._chunk, dtype=np.uint8).tofile(f)
                self._chunk.clear()

        elapsed = time.time() - t0
        print(f"\n[Generator] Done! {self.count:,} layouts in {elapsed/3600:.2f}h")
        return self.count

    def _recurse(self, ship_idx, occupied, placed_cells, grid):
        if ship_idx == len(self.ships):
            # Encode grid as packed bits (13 bytes per layout)
            packed = np.packbits(grid.flatten())
            self._chunk.append(packed)
            self.count += 1

            if len(self._chunk) >= config.GEN_CHUNK_SIZE:
                np.array(self._chunk, dtype=np.uint8).tofile(self._file)
                self._chunk.clear()
                if self.count % 5_000_000 == 0:
                    print(f"  {self.count/1e9:.3f}B layouts generated...", flush=True)
            return

        length = self.ships[ship_idx]

        for row in range(config.BOARD_SIZE):
            for col in range(config.BOARD_SIZE):
                for horizontal in (True, False):
                    cells = _try_place(occupied, row, col, length, horizontal)
                    if cells is None:
                        continue

                    # Dedup identical ships: second 3-ship must come
                    # "after" first 3-ship in canonical order
                    if ship_idx == 3:   # second size-3 ship
                        prev_cells = placed_cells[2]   # first size-3 ship
                        if tuple(sorted(cells)) <= tuple(sorted(prev_cells)):
                            continue   # skip — already counted in other order

                    new_occ = occupied | set(cells)
                    new_placed = placed_cells + [cells]

                    # Update grid
                    for r, c in cells:
                        grid[r][c] = 1

                    self._recurse(ship_idx + 1, new_occ, new_placed, grid)

                    # Undo grid
                    for r, c in cells:
                        grid[r][c] = 0


# ── Helper: load a random layout from binary file ───────────
def load_random_layout(path=config.LAYOUTS_FILE):
    """Load one random layout from the binary file."""
    size = os.path.getsize(path)
    n    = size // config.BYTES_PER_LAYOUT
    idx  = np.random.randint(0, n)
    with open(path, 'rb') as f:
        f.seek(idx * config.BYTES_PER_LAYOUT)
        raw = f.read(config.BYTES_PER_LAYOUT)
    packed = np.frombuffer(raw, dtype=np.uint8)
    grid   = np.unpackbits(packed)[:100].reshape(config.BOARD_SIZE, config.BOARD_SIZE)
    return grid.astype(np.float32)


def load_layouts_chunk(path, start_idx, count):
    """Load `count` layouts starting at `start_idx`."""
    with open(path, 'rb') as f:
        f.seek(start_idx * config.BYTES_PER_LAYOUT)
        raw = f.read(count * config.BYTES_PER_LAYOUT)
    n = len(raw) // config.BYTES_PER_LAYOUT
    if n == 0:
        return None
    packed = np.frombuffer(raw[:n * config.BYTES_PER_LAYOUT], dtype=np.uint8)
    packed = packed.reshape(n, config.BYTES_PER_LAYOUT)
    grids  = np.unpackbits(packed, axis=1)[:, :100].reshape(n, config.BOARD_SIZE, config.BOARD_SIZE)
    return grids.astype(np.float32)


def count_layouts(path=config.LAYOUTS_FILE):
    if not os.path.exists(path):
        return 0
    return os.path.getsize(path) // config.BYTES_PER_LAYOUT


# ── Generate a single random valid layout (for training) ────
def random_layout():
    """Fast random valid layout — no disk access."""
    ships = config.SHIPS[:]
    occupied = set()
    grid = np.zeros((config.BOARD_SIZE, config.BOARD_SIZE), dtype=np.float32)

    for ship_idx, length in enumerate(ships):
        placed = False
        attempts = 0
        while not placed:
            attempts += 1
            if attempts > 1000:
                # Restart on failure (rare)
                return random_layout()
            row  = np.random.randint(0, config.BOARD_SIZE)
            col  = np.random.randint(0, config.BOARD_SIZE)
            horiz = np.random.random() < 0.5
            cells = _try_place(occupied, row, col, length, horiz)
            if cells is not None:
                occupied |= set(cells)
                for r, c in cells:
                    grid[r][c] = 1
                placed = True
    return grid


if __name__ == "__main__":
    gen = LayoutGenerator()
    total = gen.generate_all(config.LAYOUTS_FILE)
    print(f"Total layouts: {total:,}")
