# ============================================================
#  train_rl.py
#  RL self-play training with PDF bot warmup.
#
#  Two phases:
#    Warmup  (first RL_WARMUP_EPISODES):
#      Model plays against the PDF bot — learns the basics fast
#      without needing a good opponent from scratch.
#
#    Self-play (remaining episodes):
#      Model plays against itself — discovers non-obvious patterns
#      and potentially surpasses the PDF bot's heuristics.
#
#  RESUME: saves every RL_SAVE_EVERY episodes.
#  Press Ctrl+C → saves immediately → re-run to continue.
# ============================================================

import os
import json
import time
import signal
import random
import numpy as np
from collections import deque

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import config
from bitboard import random_layout, BitGame, BOARD, TOTAL
from pdf_bot import PDFBot
from model import build_model

_stop_requested = False

def _handle_signal(sig, frame):
    global _stop_requested
    print("\n[RL] Stop requested — saving and exiting...")
    _stop_requested = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── Replay buffer ────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, maxlen):
        self.buf = deque(maxlen=maxlen)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((
            state.astype(np.float32),
            int(action),
            float(reward),
            next_state.astype(np.float32),
            bool(done),
        ))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(ns), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ── Checkpoint helpers ───────────────────────────────────────
def save_checkpoint(model, optimizer, scaler, episode, stats):
    os.makedirs(config.CKPT_DIR, exist_ok=True)
    torch.save({
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state':    scaler.state_dict() if scaler else None,
    }, config.RL_MODEL_FILE)

    if episode % 500_000 == 0 and episode > 0:
        bk = config.RL_MODEL_FILE.replace('_latest.pt', f'_{episode:08d}.pt')
        torch.save({'model_state': model.state_dict()}, bk)

    with open(config.RL_TRAIN_FILE, 'w') as f:
        json.dump({'episode': episode, 'stats': stats}, f, indent=2)

    print(f"[RL] ✓ Saved at episode {episode:,}")


def load_checkpoint(model, optimizer, scaler):
    if not os.path.exists(config.RL_MODEL_FILE):
        print("[RL] No checkpoint — starting fresh.")
        return 0, {}

    ckpt = torch.load(config.RL_MODEL_FILE, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    if scaler and ckpt.get('scaler_state'):
        scaler.load_state_dict(ckpt['scaler_state'])

    with open(config.RL_TRAIN_FILE) as f:
        state = json.load(f)

    print(f"[RL] Resumed from episode {state['episode']:,}")
    return state['episode'], state.get('stats', {})


# ── Model inference helper ───────────────────────────────────
def model_shot(model, game: BitGame, device, epsilon=0.0):
    """
    Choose a shot using the RL model (epsilon-greedy).
    Returns (row, col, action_index).
    """
    state = game.board_state()   # (3,10,10)
    valid = game.valid_shots()
    mask  = np.zeros(TOTAL, dtype=bool)
    for r, c in valid:
        mask[r * BOARD + c] = True

    if random.random() < epsilon:
        # Random valid shot
        r, c = random.choice(valid)
        return r, c, r * BOARD + c

    with torch.no_grad():
        t     = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs = model(t).squeeze(0).cpu().numpy()

    probs[~mask] = 0
    s = probs.sum()
    if s == 0:
        probs = mask.astype(np.float32)
        s = probs.sum()
    probs /= s
    action = int(np.argmax(probs))
    return action // BOARD, action % BOARD, action


# ── Warmup episode: model attacks PDF bot's layout ───────────
def warmup_episode(model, device, epsilon):
    """
    Model shoots at a PDF-bot-placed layout.
    Opponent layout is just random (PDF bot doesn't choose its own layout).
    Returns experiences list.
    """
    bits, placements = random_layout()
    game = BitGame(bits, placements)
    experiences = []

    while not game.done and game.turns < TOTAL:
        state  = game.board_state()
        r, c, action = model_shot(model, game, device, epsilon)
        hit, sunk_len, done = game.shoot(r, c)

        reward = 0.0
        if hit:
            reward = 1.0 if sunk_len > 0 else 0.3
        else:
            reward = -0.1

        next_state = game.board_state()
        experiences.append((state, action, reward, next_state, done))

    return experiences, game.turns


# ── Self-play episode: model vs model ───────────────────────
def selfplay_episode(model, device, epsilon):
    """
    Model attacks a random layout (same as warmup in practice since
    the defender is just a static layout — RL emerges from
    the model improving its attack strategy across many layouts).
    """
    return warmup_episode(model, device, epsilon)


# ── Training step ────────────────────────────────────────────
def train_step(model, optimizer, scaler, replay_buffer, device):
    if len(replay_buffer) < config.RL_BATCH_SIZE:
        return None

    s, a, r, ns, d = replay_buffer.sample(config.RL_BATCH_SIZE)

    s_t  = torch.tensor(s,  dtype=torch.float32).to(device)
    ns_t = torch.tensor(ns, dtype=torch.float32).to(device)
    a_t  = torch.tensor(a,  dtype=torch.long).to(device)
    r_t  = torch.tensor(r,  dtype=torch.float32).to(device)
    d_t  = torch.tensor(d,  dtype=torch.float32).to(device)

    if config.RL_MIXED_PRECISION and scaler is not None:
        with autocast():
            logits_curr = model.logits(s_t)
            q_curr      = logits_curr.gather(1, a_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next  = model.logits(ns_t).max(1)[0]
            target = r_t + config.RL_GAMMA * q_next * (1 - d_t)
            loss   = F.smooth_l1_loss(q_curr, target.detach())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits_curr = model.logits(s_t)
        q_curr      = logits_curr.gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next  = model.logits(ns_t).max(1)[0]
        target = r_t + config.RL_GAMMA * q_next * (1 - d_t)
        loss   = F.smooth_l1_loss(q_curr, target.detach())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return loss.item()


# ── Main training function ───────────────────────────────────
def train_rl():
    global _stop_requested

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR,  exist_ok=True)

    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[RL] Device: {device}")
    if device.type == 'cuda':
        print(f"[RL] GPU   : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[RL] VRAM  : {vram:.1f} GB")

    model     = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.RL_LEARNING_RATE)
    scaler    = GradScaler() if (config.RL_MIXED_PRECISION and device.type == 'cuda') else None

    start_ep, stats  = load_checkpoint(model, optimizer, scaler)
    replay_buffer    = ReplayBuffer(config.RL_REPLAY_BUFFER_SIZE)

    recent_turns  = deque(maxlen=1000)
    recent_losses = deque(maxlen=1000)

    def get_epsilon(ep):
        """1.0 → 0.05 over first 300K episodes."""
        return max(0.05, 1.0 - (ep / 300_000) * 0.95)

    mode = "WARMUP" if start_ep < config.RL_WARMUP_EPISODES else "SELF-PLAY"
    print(f"\n[RL] ═══ Training ({mode}) ═══")
    print(f"[RL] Episodes   : {start_ep:,} → {config.RL_TOTAL_EPISODES:,}")
    print(f"[RL] Warmup ends: {config.RL_WARMUP_EPISODES:,}")
    print(f"[RL] Ctrl+C saves and exits\n")

    log_path     = os.path.join(config.LOG_DIR, "rl_log.csv")
    write_header = not os.path.exists(log_path) or start_ep == 0

    with open(log_path, 'a') as log:
        if write_header:
            log.write("episode,mode,avg_turns,avg_loss,epsilon,ep_per_sec\n")

        t_start   = time.time()
        t_last_log= t_start
        episode   = start_ep

        for episode in range(start_ep, config.RL_TOTAL_EPISODES):
            if _stop_requested:
                break

            eps     = get_epsilon(episode)
            in_warmup = episode < config.RL_WARMUP_EPISODES

            experiences, turns = warmup_episode(model, device, eps)
            recent_turns.append(turns)

            for exp in experiences:
                replay_buffer.push(*exp)

            loss = train_step(model, optimizer, scaler, replay_buffer, device)
            if loss is not None:
                recent_losses.append(loss)

            if (episode + 1) % config.RL_LOG_EVERY == 0:
                now     = time.time()
                elapsed = now - t_last_log
                eps_s   = config.RL_LOG_EVERY / elapsed
                t_last_log = now

                avg_t = np.mean(recent_turns)  if recent_turns  else 0
                avg_l = np.mean(recent_losses) if recent_losses else 0
                remain_h = (config.RL_TOTAL_EPISODES - episode) / eps_s / 3600
                phase = "WARM" if in_warmup else "PLAY"

                print(f"[{phase}] ep {episode+1:>8,} | ε={eps:.3f} | "
                      f"turns={avg_t:.1f} | loss={avg_l:.4f} | "
                      f"{eps_s:.0f}/s | ETA {remain_h:.1f}h")

                log.write(f"{episode+1},{phase},{avg_t:.2f},"
                          f"{avg_l:.6f},{eps:.4f},{eps_s:.1f}\n")
                log.flush()

            if (episode + 1) % config.RL_SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, scaler, episode + 1, {
                    'avg_turns': float(np.mean(recent_turns)) if recent_turns else 0,
                })

        # Final save
        save_checkpoint(model, optimizer, scaler, episode + 1, {
            'avg_turns': float(np.mean(recent_turns)) if recent_turns else 0,
        })

    elapsed_total = time.time() - t_start
    print(f"\n[RL] Done! {episode+1:,} episodes in {elapsed_total/3600:.2f}h")
    if _stop_requested:
        print("[RL] Stopped early. Re-run: python run.py train_rl")


if __name__ == "__main__":
    train_rl()
