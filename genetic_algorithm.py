# ============================================================
#  genetic_algorithm.py
#  Genetic Algorithm optimizer using the PDF bot as fitness judge.
#
#  Each "individual" is a list of ship placements:
#    [(row, col, length, horizontal), ...]   ← one per ship
#
#  Operations:
#    Selection : Keep top GA_SELECT_FRAC by fitness score
#    Elitism   : Top GA_ELITE_FRAC survive unchanged
#    Crossover : Take ships from two parents, resolve overlaps
#    Mutation  : Randomly move one ship
#
#  RESUME SUPPORT:
#    Every GA_SAVE_EVERY generations → saves population + scores
#    On restart → loads last checkpoint and continues
# ============================================================

import os
import json
import time
import signal
import random
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

import config
from bitboard import (random_layout, ship_bits, encode_layout,
                      bits_to_grid, layout_to_grid, BOARD, TOTAL)
from pdf_bot import score_layout_worker

_stop_requested = False

def _handle_signal(sig, frame):
    global _stop_requested
    print("\n[GA] Stop requested — saving after this generation...")
    _stop_requested = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── Individual encoding ──────────────────────────────────────
# An individual = list of (row, col, length, horizontal) tuples
# Stored as list-of-lists for JSON serialization

def individual_to_json(ind):
    return [[r, c, l, int(h)] for r, c, l, h in ind]

def individual_from_json(data):
    return [(r, c, l, bool(h)) for r, c, l, h in data]

def individual_to_bits(ind):
    result = encode_layout(ind)
    if result is None:
        return None, ind
    bits, _ = result
    return bits, ind


# ── Crossover ────────────────────────────────────────────────
def crossover(parent_a, parent_b):
    """
    Produce a child by taking each ship from either parent A or B.
    Resolves overlaps by retrying the conflicting ship from the other parent
    or generating a random valid placement.

    Returns a valid list of placements, or None if failed.
    """
    n_ships = len(parent_a)

    for _attempt in range(config.GA_CROSSOVER_MAX_RETRIES):
        combined  = 0
        child_ind = []
        ok = True

        for i in range(n_ships):
            # Randomly pick which parent to try first
            if random.random() < 0.5:
                primary, fallback = parent_a[i], parent_b[i]
            else:
                primary, fallback = parent_b[i], parent_a[i]

            row, col, length, horiz = primary
            b = ship_bits(row, col, length, horiz)

            if b and not (combined & b):
                combined  |= b
                child_ind.append(primary)
                continue

            # Try fallback parent
            row, col, length, horiz = fallback
            b = ship_bits(row, col, length, horiz)
            if b and not (combined & b):
                combined  |= b
                child_ind.append(fallback)
                continue

            # Both parents conflict — try random placement for this ship
            placed = False
            for _inner in range(50):
                r = random.randint(0, BOARD - 1)
                c = random.randint(0, BOARD - 1)
                h = random.random() < 0.5
                b = ship_bits(r, c, length, h)
                if b and not (combined & b):
                    combined |= b
                    child_ind.append((r, c, length, h))
                    placed = True
                    break

            if not placed:
                ok = False
                break

        if ok:
            return child_ind

    return None  # couldn't produce valid child


# ── Mutation ─────────────────────────────────────────────────
def mutate(individual):
    """
    Randomly move one ship to a new valid position.
    Returns mutated individual (modifies a copy).
    """
    ind = list(individual)  # copy
    ship_idx = random.randint(0, len(ind) - 1)
    _, _, length, _ = ind[ship_idx]

    # Remove chosen ship from occupied set
    combined = 0
    for i, (r, c, l, h) in enumerate(ind):
        if i != ship_idx:
            combined |= ship_bits(r, c, l, h)

    # Find a new valid placement
    for _ in range(200):
        r = random.randint(0, BOARD - 1)
        c = random.randint(0, BOARD - 1)
        h = random.random() < 0.5
        b = ship_bits(r, c, length, h)
        if b and not (combined & b):
            ind[ship_idx] = (r, c, length, h)
            return ind

    return ind  # couldn't mutate — return unchanged


# ── Population initialization ────────────────────────────────
def init_population(n):
    """Generate n random valid individuals."""
    pop = []
    for _ in tqdm(range(n), desc="Init population", unit="layout"):
        _, placements = random_layout()
        pop.append(placements)
    return pop


# ── Fitness evaluation (parallel) ────────────────────────────
def evaluate_population(population, n_games, workers):
    """
    Score all individuals in population using multiprocessing.
    Returns numpy array of scores (shape: N).
    """
    args = []
    for ind in population:
        bits_result = encode_layout(ind)
        if bits_result is None:
            # Invalid layout — penalize
            args.append((0, ind, n_games))
        else:
            bits, _ = bits_result
            args.append((bits, ind, n_games))

    if workers > 1:
        with Pool(workers) as pool:
            scores = list(tqdm(
                pool.imap(score_layout_worker, args, chunksize=50),
                total=len(args),
                desc="Fitness eval",
                unit="layout"
            ))
    else:
        scores = [score_layout_worker(a) for a in tqdm(args, desc="Fitness eval")]

    return np.array(scores, dtype=np.float32)


# ── Checkpoint I/O ────────────────────────────────────────────
def save_checkpoint(generation, population, scores, elapsed_time):
    os.makedirs(config.CKPT_DIR, exist_ok=True)

    # Save population as JSON (placements are small)
    pop_json = [individual_to_json(ind) for ind in population]

    state = {
        'generation':    generation,
        'elapsed_time':  elapsed_time,
        'best_score':    float(scores.max()),
        'mean_score':    float(scores.mean()),
        'population':    pop_json,
        'scores':        scores.tolist(),
    }

    # Write to temp file first, then rename (atomic — survives crashes)
    tmp = config.GA_CKPT_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(state, f)
    os.replace(tmp, config.GA_CKPT_FILE)

    print(f"[GA] ✓ Checkpoint saved at generation {generation}")


def load_checkpoint():
    """Returns (generation, population, scores, elapsed) or (0, None, None, 0)."""
    if not os.path.exists(config.GA_CKPT_FILE):
        return 0, None, None, 0.0

    try:
        with open(config.GA_CKPT_FILE) as f:
            state = json.load(f)

        generation = state['generation']
        population = [individual_from_json(ind) for ind in state['population']]
        scores     = np.array(state['scores'], dtype=np.float32)
        elapsed    = state.get('elapsed_time', 0.0)

        print(f"[GA] Resumed from generation {generation}/{config.GA_GENERATIONS}")
        print(f"[GA] Best score: {scores.max():.2f} | Mean: {scores.mean():.2f}")
        return generation, population, scores, elapsed
    except Exception as e:
        print(f"[GA] Warning: checkpoint load failed ({e}) — starting fresh")
        return 0, None, None, 0.0


# ── Main GA loop ─────────────────────────────────────────────
def run_ga():
    global _stop_requested

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR,  exist_ok=True)

    # Resume or fresh start
    start_gen, population, scores, prev_elapsed = load_checkpoint()

    if population is None:
        print(f"\n[GA] ═══ Starting GA (fresh) ═══")
        print(f"  Population : {config.GA_POPULATION:,}")
        print(f"  Generations: {config.GA_GENERATIONS}")
        print(f"  Fitness    : {config.GA_FITNESS_GAMES} PDF games per layout")
        print(f"  CPU workers: {config.GA_WORKERS}\n")

        print("[GA] Initializing population...")
        population = init_population(config.GA_POPULATION)
        print("[GA] Evaluating initial fitness...")
        scores = evaluate_population(population, config.GA_FITNESS_GAMES, config.GA_WORKERS)
        save_checkpoint(0, population, scores, 0.0)
    else:
        print(f"\n[GA] ═══ Resuming GA from generation {start_gen} ═══\n")

    log_path = os.path.join(config.LOG_DIR, "ga_log.csv")
    write_header = not os.path.exists(log_path) or start_gen == 0

    t_start = time.time()

    with open(log_path, 'a') as log:
        if write_header:
            log.write("generation,best_score,mean_score,top10_mean,elapsed_h\n")

        for gen in range(start_gen + 1, config.GA_GENERATIONS + 1):
            if _stop_requested:
                break

            t_gen = time.time()

            N         = len(population)
            elite_n   = max(1, int(N * config.GA_ELITE_FRAC))
            select_n  = max(elite_n + 1, int(N * config.GA_SELECT_FRAC))

            # ── Selection ────────────────────────────────────
            ranked_idx = np.argsort(scores)[::-1]    # highest score first
            elite_idx  = ranked_idx[:elite_n]
            parent_idx = ranked_idx[:select_n]

            elites  = [population[i] for i in elite_idx]
            parents = [population[i] for i in parent_idx]

            # ── Breed new generation ──────────────────────────
            children = list(elites)   # elites survive unchanged

            while len(children) < N:
                pa = random.choice(parents)
                pb = random.choice(parents)

                child = crossover(pa, pb)
                if child is None:
                    # crossover failed — clone best parent with mutation
                    child = list(pa)

                # Mutate with probability
                if random.random() < config.GA_MUTATE_PROB:
                    child = mutate(child)

                children.append(child)

            population = children[:N]

            # ── Evaluate new generation ───────────────────────
            # Elites keep their old scores (faster)
            new_scores = np.zeros(N, dtype=np.float32)
            for i in range(elite_n):
                new_scores[i] = scores[elite_idx[i]]

            non_elite_pop    = population[elite_n:]
            non_elite_scores = evaluate_population(
                non_elite_pop, config.GA_FITNESS_GAMES, config.GA_WORKERS
            )
            new_scores[elite_n:] = non_elite_scores
            scores = new_scores

            # ── Stats ─────────────────────────────────────────
            best_score   = scores.max()
            mean_score   = scores.mean()
            top10_mean   = np.sort(scores)[-10:].mean()
            gen_time     = time.time() - t_gen
            total_elapsed= prev_elapsed + (time.time() - t_start)
            eta_h        = (config.GA_GENERATIONS - gen) * gen_time / 3600

            print(f"Gen {gen:>4}/{config.GA_GENERATIONS} | "
                  f"best={best_score:.1f} | mean={mean_score:.1f} | "
                  f"top10={top10_mean:.1f} | "
                  f"{gen_time:.0f}s/gen | ETA {eta_h:.1f}h")

            log.write(f"{gen},{best_score:.2f},{mean_score:.2f},"
                      f"{top10_mean:.2f},{total_elapsed/3600:.3f}\n")
            log.flush()

            # ── Checkpoint ────────────────────────────────────
            if gen % config.GA_SAVE_EVERY == 0 or _stop_requested:
                save_checkpoint(gen, population, scores, total_elapsed)

    # ── Final save ────────────────────────────────────────────
    final_gen = gen
    save_checkpoint(final_gen, population, scores, prev_elapsed + (time.time() - t_start))

    # ── Extract top 10K and save as grids ────────────────────
    print(f"\n[GA] Extracting top {config.GA_POPULATION:,} layouts...")
    ranked = np.argsort(scores)[::-1]
    top_layouts = []
    top_scores  = []
    top_placements = []

    for i in ranked:
        ind   = population[i]
        result = encode_layout(ind)
        if result is None:
            continue
        bits, _ = result
        grid = bits_to_grid(bits)
        top_layouts.append(grid)
        top_scores.append(scores[i])
        top_placements.append(individual_to_json(ind))

    top_layouts = np.array(top_layouts, dtype=np.float32)
    top_scores  = np.array(top_scores,  dtype=np.float32)

    os.makedirs(config.TOP_DIR, exist_ok=True)
    np.save(os.path.join(config.CKPT_DIR, "ga_top_layouts.npy"),    top_layouts)
    np.save(os.path.join(config.CKPT_DIR, "ga_top_scores.npy"),     top_scores)

    with open(os.path.join(config.CKPT_DIR, "ga_top_placements.json"), 'w') as f:
        json.dump(top_placements, f)

    print(f"\n[GA] ═══ GA Complete ═══")
    print(f"  Generations run : {final_gen}")
    print(f"  Best score      : {top_scores[0]:.2f} shots survived")
    print(f"  Mean (top 100)  : {top_scores[:100].mean():.2f} shots survived")
    print(f"  Saved to        : {config.CKPT_DIR}/ga_top_layouts.npy")

    if _stop_requested:
        print("[GA] Stopped early. Re-run to continue from last checkpoint.")

    return top_layouts, top_scores


if __name__ == "__main__":
    run_ga()
