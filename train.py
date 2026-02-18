# ============================================================
#  train.py
#  Self-play RL training with full stop/resume support.
#
#  HOW RESUME WORKS:
#   - Every SAVE_EVERY episodes, saves:
#       checkpoints/model_latest.pt     ← model weights
#       checkpoints/model_NNNNNN.pt     ← timestamped backup
#       checkpoints/train_state.json    ← episode count, stats, etc.
#   - On startup, if these files exist, training resumes exactly
#     where it left off.
#   - Press Ctrl+C to stop cleanly — it saves before exiting.
# ============================================================

import os
import json
import time
import signal
import random
import numpy as np
from collections import deque
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import config
from model import build_model
from layout_generator import random_layout
from game_engine import BattleshipGame, hunt_target_shot

# ── Global flag for clean Ctrl+C shutdown ───────────────────
_stop_requested = False

def _handle_signal(sig, frame):
    global _stop_requested
    print("\n[Train] Stop requested — will save and exit after this episode...")
    _stop_requested = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── Replay buffer ────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            state.astype(np.float32),
            int(action),
            float(reward),
            next_state.astype(np.float32),
            bool(done)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── Checkpoint helpers ───────────────────────────────────────
def save_checkpoint(model, optimizer, scaler, episode, stats, path=config.MODEL_FILE):
    os.makedirs(config.CKPT_DIR, exist_ok=True)

    torch.save({
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scaler_state':    scaler.state_dict() if scaler else None,
    }, path)

    # Timestamped backup every 500K episodes
    if episode % 500_000 == 0 and episode > 0:
        backup = os.path.join(config.CKPT_DIR, f"model_{episode:08d}.pt")
        torch.save({'model_state': model.state_dict()}, backup)

    state_dict = {
        'episode': episode,
        'stats':   stats,
    }
    with open(config.TRAIN_CKPT_FILE, 'w') as f:
        json.dump(state_dict, f, indent=2)

    print(f"[Train] ✓ Checkpoint saved at episode {episode:,}")


def load_checkpoint(model, optimizer, scaler):
    """Returns (start_episode, stats) or (0, {}) if no checkpoint."""
    if not os.path.exists(config.MODEL_FILE):
        print("[Train] No checkpoint found — starting from scratch.")
        return 0, {}

    print(f"[Train] Loading checkpoint: {config.MODEL_FILE}")
    ckpt = torch.load(config.MODEL_FILE, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    if scaler and ckpt.get('scaler_state'):
        scaler.load_state_dict(ckpt['scaler_state'])

    with open(config.TRAIN_CKPT_FILE) as f:
        state = json.load(f)

    ep    = state['episode']
    stats = state.get('stats', {})
    print(f"[Train] Resumed from episode {ep:,}")
    return ep, stats


# ── One training step (DQN-style) ───────────────────────────
def train_step(model, optimizer, scaler, replay_buffer, device):
    if len(replay_buffer) < config.BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(config.BATCH_SIZE)

    states_t      = torch.tensor(states,      dtype=torch.float32).to(device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32).to(device)
    actions_t     = torch.tensor(actions,     dtype=torch.long).to(device)
    rewards_t     = torch.tensor(rewards,     dtype=torch.float32).to(device)
    dones_t       = torch.tensor(dones,       dtype=torch.float32).to(device)

    if config.MIXED_PRECISION and scaler is not None:
        with autocast():
            logits = model.action_logits(states_t)
            q_vals = logits.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_logits = model.action_logits(next_states_t)
                next_q      = next_logits.max(1)[0]

            target = rewards_t + config.GAMMA * next_q * (1 - dones_t)
            loss   = F.smooth_l1_loss(q_vals, target.detach())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits = model.action_logits(states_t)
        q_vals = logits.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_logits = model.action_logits(next_states_t)
            next_q      = next_logits.max(1)[0]

        target = rewards_t + config.GAMMA * next_q * (1 - dones_t)
        loss   = F.smooth_l1_loss(q_vals, target.detach())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return loss.item()


# ── One self-play episode ────────────────────────────────────
def run_episode(model, device, epsilon):
    """
    Play one game using epsilon-greedy policy.
    Returns list of (state, action, reward, next_state, done) tuples.
    """
    layout_a = random_layout()
    layout_b = random_layout()

    game = BattleshipGame(layout_b)   # model attacks layout_b
    experiences = []

    while not game.done:
        state = game.state()
        mask  = game.valid_shot_mask()

        # Epsilon-greedy: explore vs exploit
        if random.random() < epsilon:
            # Random valid shot
            valid = np.where(mask)[0]
            action = int(np.random.choice(valid))
        else:
            with torch.no_grad():
                t     = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                probs = model(t).squeeze(0).cpu().numpy()
            probs[~mask] = 0
            if probs.sum() == 0:
                probs = mask.astype(np.float32)
            probs /= probs.sum()
            action = int(np.argmax(probs))   # greedy

        r, c = divmod(action, config.BOARD_SIZE)
        _, _, done, reward = game.shoot(r, c)
        next_state = game.state()

        experiences.append((state, action, reward, next_state, done))

        # Safety: avoid infinite loops
        if game.turns >= 100:
            break

    return experiences, game.turns


# ── Main training function ───────────────────────────────────
def train():
    global _stop_requested

    os.makedirs(config.CKPT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR,  exist_ok=True)

    # Device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    if device.type == 'cuda':
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Train] VRAM: {torch.cuda.get_device_properties(0).total_memory // 1e6:.0f} MB")

    # Model + optimizer + scaler
    model     = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler    = GradScaler() if (config.MIXED_PRECISION and device.type == 'cuda') else None

    # Resume from checkpoint if available
    start_ep, stats = load_checkpoint(model, optimizer, scaler)

    # Replay buffer
    replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)

    # Stats tracking
    recent_turns  = deque(maxlen=1000)
    recent_losses = deque(maxlen=1000)
    total_episodes = stats.get('total_episodes', start_ep)

    # Epsilon schedule: 1.0 → 0.05 over first 500K episodes
    def get_epsilon(ep):
        return max(0.05, 1.0 - (ep / 500_000) * 0.95)

    print(f"\n[Train] ═══ Training {'(resumed)' if start_ep > 0 else '(fresh)'} ═══")
    print(f"[Train] Episodes: {start_ep:,} → {config.TOTAL_EPISODES:,}")
    print(f"[Train] Batch size: {config.BATCH_SIZE}")
    print(f"[Train] Mixed precision: {config.MIXED_PRECISION}")
    print(f"[Train] Saving every: {config.SAVE_EVERY:,} episodes")
    print(f"[Train] Press Ctrl+C to stop and save\n")

    log_file = os.path.join(config.LOG_DIR, "train_log.csv")
    write_header = not os.path.exists(log_file)

    with open(log_file, 'a') as log:
        if write_header:
            log.write("episode,avg_turns,avg_loss,epsilon,eps_per_sec\n")

        t_start = time.time()
        t_last_log = t_start

        for episode in range(start_ep, config.TOTAL_EPISODES):
            if _stop_requested:
                break

            epsilon = get_epsilon(episode)

            # Run one episode
            experiences, turns = run_episode(model, device, epsilon)
            recent_turns.append(turns)

            # Push to replay buffer
            for exp in experiences:
                replay_buffer.push(*exp)

            # Train every step
            loss = train_step(model, optimizer, scaler, replay_buffer, device)
            if loss is not None:
                recent_losses.append(loss)

            # ── Logging ─────────────────────────────────────
            if (episode + 1) % config.LOG_EVERY == 0:
                now         = time.time()
                elapsed     = now - t_last_log
                eps_per_sec = config.LOG_EVERY / elapsed
                t_last_log  = now

                avg_turns = np.mean(recent_turns)
                avg_loss  = np.mean(recent_losses) if recent_losses else 0

                remaining = (config.TOTAL_EPISODES - episode) / eps_per_sec
                eta_h = remaining / 3600

                print(f"Ep {episode+1:>8,} | "
                      f"ε={epsilon:.3f} | "
                      f"turns={avg_turns:.1f} | "
                      f"loss={avg_loss:.4f} | "
                      f"buf={len(replay_buffer):>6,} | "
                      f"{eps_per_sec:.0f} ep/s | "
                      f"ETA {eta_h:.1f}h")

                log.write(f"{episode+1},{avg_turns:.2f},{avg_loss:.6f},{epsilon:.4f},{eps_per_sec:.1f}\n")
                log.flush()

            # ── Checkpoint ──────────────────────────────────
            if (episode + 1) % config.SAVE_EVERY == 0:
                save_checkpoint(model, optimizer, scaler, episode + 1, {
                    'total_episodes': episode + 1,
                    'avg_turns':      float(np.mean(recent_turns)) if recent_turns else 0,
                })

        # ── Final save ──────────────────────────────────────
        final_ep = min(episode + 1, config.TOTAL_EPISODES)
        save_checkpoint(model, optimizer, scaler, final_ep, {
            'total_episodes': final_ep,
            'avg_turns':      float(np.mean(recent_turns)) if recent_turns else 0,
        })

        total_time = time.time() - t_start
        print(f"\n[Train] ✓ Training complete! {final_ep:,} episodes in {total_time/3600:.2f}h")

        if _stop_requested:
            print("[Train] Stopped early by user. Resume with: python run.py train")


if __name__ == "__main__":
    train()
