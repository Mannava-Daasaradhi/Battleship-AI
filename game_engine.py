# ============================================================
#  game_engine.py
#  Fast Battleship game simulation (pure NumPy, no Python loops
#  in the hot path).  Supports batched simulation for GPU scoring.
# ============================================================

import numpy as np
import config

BOARD = config.BOARD_SIZE
SHIPS = config.SHIPS


# ── Ship cell tracking ───────────────────────────────────────
def get_ship_cells(grid):
    """
    Returns a list of frozensets, one per ship.
    Identifies connected components of 1s (ships).
    Simple flood-fill.
    """
    visited = np.zeros_like(grid, dtype=bool)
    ships   = []

    for r in range(BOARD):
        for c in range(BOARD):
            if grid[r, c] == 1 and not visited[r, c]:
                # BFS
                cells = []
                q = [(r, c)]
                visited[r, c] = True
                while q:
                    cr, cc = q.pop()
                    cells.append((cr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < BOARD and 0 <= nc < BOARD:
                            if grid[nr, nc] == 1 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                ships.append(frozenset(cells))
    return ships


# ── Single Game ──────────────────────────────────────────────
class BattleshipGame:
    """
    Simulates one game between an attacker (model/policy) and a defender
    (static layout).  Returns game log for RL training.

      board state encoding:
        0 = unknown
        1 = hit
       -1 = miss
    """

    def __init__(self, defender_layout: np.ndarray):
        self.layout     = defender_layout           # (10,10) binary
        self.ship_cells = get_ship_cells(defender_layout)
        self.ship_sunk  = [False] * len(self.ship_cells)
        self.board      = np.zeros((BOARD, BOARD), dtype=np.float32)
        self.shots_fired= set()
        self.turns      = 0
        self.done       = False
        self.total_ship_cells = int(defender_layout.sum())
        self.hits_so_far      = 0

    def state(self):
        """
        Returns (3, 10, 10) tensor:
          channel 0: unknown cells (1 where not yet shot)
          channel 1: hits
          channel 2: misses
        """
        unknown = (self.board == 0).astype(np.float32)
        hits    = (self.board == 1).astype(np.float32)
        misses  = (self.board == -1).astype(np.float32)
        return np.stack([unknown, hits, misses], axis=0)

    def shoot(self, row, col):
        """
        Fire at (row, col).
        Returns (hit: bool, sunk: bool, done: bool, reward: float)
        """
        assert (row, col) not in self.shots_fired, "Already shot here!"
        self.shots_fired.add((row, col))
        self.turns += 1

        hit = self.layout[row, col] == 1

        if hit:
            self.board[row, col] = 1
            self.hits_so_far += 1

            # Check if any ship sunk
            sunk = False
            for i, ship in enumerate(self.ship_cells):
                if not self.ship_sunk[i] and (row, col) in ship:
                    if all(self.board[r, c] == 1 for r, c in ship):
                        self.ship_sunk[i] = True
                        sunk = True

            if self.hits_so_far == self.total_ship_cells:
                self.done = True

            reward = 1.0 if sunk else 0.3
        else:
            self.board[row, col] = -1
            reward = -0.1

        return hit, any(self.ship_sunk), self.done, reward

    def valid_shots(self):
        """Returns list of (r,c) not yet shot."""
        return [(r, c) for r in range(BOARD) for c in range(BOARD)
                if (r, c) not in self.shots_fired]

    def valid_shot_mask(self):
        """Returns (100,) bool mask — True where shooting is valid."""
        mask = np.ones(BOARD * BOARD, dtype=bool)
        for r, c in self.shots_fired:
            mask[r * BOARD + c] = False
        return mask


# ── Self-play: two layouts, two models ───────────────────────
def play_one_game(layout_a, layout_b, model_a, model_b, device):
    """
    Full self-play game. Alternating turns.
    Returns:
      winner       : 'A' or 'B'
      turns_a      : how many turns A needed to win (or max_turns)
      experiences_a: list of (state, action, reward, next_state, done)
      experiences_b: same for B
    """
    import torch

    game_a = BattleshipGame(layout_b)  # A shoots at B's layout
    game_b = BattleshipGame(layout_a)  # B shoots at A's layout

    exp_a, exp_b = [], []
    MAX_TURNS = 100  # safety cap (100 shots each)

    for _ in range(MAX_TURNS):
        # ── A's turn ────────────────────────────────────────
        if not game_a.done:
            state_a = game_a.state()
            mask_a  = game_a.valid_shot_mask()

            with torch.no_grad():
                t = torch.tensor(state_a, dtype=torch.float32).unsqueeze(0).to(device)
                probs = model_a(t).squeeze(0).cpu().numpy()

            # Mask invalid shots, renormalize
            probs[~mask_a] = 0
            if probs.sum() == 0:
                probs = mask_a.astype(np.float32)
            probs /= probs.sum()

            action = np.random.choice(BOARD * BOARD, p=probs)
            r, c   = divmod(action, BOARD)
            hit, sunk, done_a, reward = game_a.shoot(r, c)
            next_state_a = game_a.state()

            exp_a.append((state_a, action, reward, next_state_a, done_a))

            if done_a:
                return 'A', game_a.turns, exp_a, exp_b

        # ── B's turn ────────────────────────────────────────
        if not game_b.done:
            state_b = game_b.state()
            mask_b  = game_b.valid_shot_mask()

            with torch.no_grad():
                t = torch.tensor(state_b, dtype=torch.float32).unsqueeze(0).to(device)
                probs = model_b(t).squeeze(0).cpu().numpy()

            probs[~mask_b] = 0
            if probs.sum() == 0:
                probs = mask_b.astype(np.float32)
            probs /= probs.sum()

            action = np.random.choice(BOARD * BOARD, p=probs)
            r, c   = divmod(action, BOARD)
            hit, sunk, done_b, reward = game_b.shoot(r, c)
            next_state_b = game_b.state()

            exp_b.append((state_b, action, reward, next_state_b, done_b))

            if done_b:
                return 'B', game_b.turns, exp_a, exp_b

    # Draw — whoever hit more wins
    winner = 'A' if game_a.hits_so_far >= game_b.hits_so_far else 'B'
    return winner, MAX_TURNS, exp_a, exp_b


# ── Batched scoring (no model, pure heuristic hunt-target) ───
def score_layouts_batch(layouts: np.ndarray, model, device, n_games: int = 20):
    """
    Score a batch of layouts by simulating them as defenders against the
    trained attacker model.  Higher score = survives more turns = better layout.

    layouts: (N, 10, 10) float32
    Returns: (N,) float32 scores
    """
    import torch
    N = len(layouts)
    total_turns = np.zeros(N, dtype=np.float32)

    for _ in range(n_games):
        for i, layout in enumerate(layouts):
            game = BattleshipGame(layout)
            while not game.done:
                state = game.state()
                mask  = game.valid_shot_mask()
                with torch.no_grad():
                    t     = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    probs = model(t).squeeze(0).cpu().numpy()
                probs[~mask] = 0
                if probs.sum() == 0:
                    probs = mask.astype(np.float32)
                probs /= probs.sum()
                action = np.random.choice(BOARD * BOARD, p=probs)
                r, c   = divmod(action, BOARD)
                game.shoot(r, c)
            total_turns[i] += game.turns

    return total_turns / n_games   # average turns survived


# ── Hunt-Target baseline AI (no model needed) ────────────────
def hunt_target_shot(board: np.ndarray):
    """
    Classic hunt-and-target heuristic.
    Used for initial training warm-up before model is good.
    """
    unknown = [(r, c) for r in range(BOARD) for c in range(BOARD)
               if board[r, c] == 0]
    hits    = [(r, c) for r in range(BOARD) for c in range(BOARD)
               if board[r, c] == 1]

    # TARGET mode: if there are hits, shoot adjacent unknowns
    if hits:
        candidates = []
        for hr, hc in hits:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = hr+dr, hc+dc
                if 0 <= nr < BOARD and 0 <= nc < BOARD and board[nr, nc] == 0:
                    candidates.append((nr, nc))
        if candidates:
            return candidates[np.random.randint(len(candidates))]

    # HUNT mode: checkerboard pattern (shoots every other cell)
    parity = [(r, c) for r, c in unknown if (r + c) % 2 == 0]
    if parity:
        return parity[np.random.randint(len(parity))]
    return unknown[np.random.randint(len(unknown))]
