# ============================================================
#  model.py
#  CNN policy network — small enough for 4GB VRAM GTX 1650.
#  Input : (3, 10, 10) board state
#  Output: (100,) shot probabilities (softmax)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class BattleshipNet(nn.Module):
    """
    Convolutional policy network for the Battleship attacker.
    
    Architecture:
      3 conv layers (spatial pattern recognition)
      → 2 FC layers
      → 100-dim softmax output (one probability per cell)

    Designed to stay under 4GB VRAM on GTX 1650.
    """

    def __init__(self):
        super().__init__()

        ch = config.CONV_CHANNELS   # [32, 64, 64]

        self.conv = nn.Sequential(
            # Conv1: detects local hit/miss patterns
            nn.Conv2d(config.INPUT_CHANNELS, ch[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),

            # Conv2: detects ship-shape patterns
            nn.Conv2d(ch[0], ch[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(inplace=True),

            # Conv3: higher-level spatial features
            nn.Conv2d(ch[1], ch[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(inplace=True),
        )

        fc_in = ch[2] * config.BOARD_SIZE * config.BOARD_SIZE   # 64*10*10 = 6400
        fc    = config.FC_DIMS                                    # [512, 256]

        self.fc = nn.Sequential(
            nn.Linear(fc_in, fc[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(fc[0], fc[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc[1], config.BOARD_SIZE * config.BOARD_SIZE),  # 100 outputs
        )

    def forward(self, x):
        """
        x: (B, 3, 10, 10) — batch of board states
        Returns: (B, 100) softmax probabilities
        """
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)   # flatten
        logits = self.fc(feat)
        return F.softmax(logits, dim=-1)

    def action_logits(self, x):
        """Returns raw logits (for loss computation)."""
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


def build_model(device):
    """Build model, print param count, move to device."""
    model = BattleshipNet().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Parameters: {params:,}  (~{params*4/1e6:.1f} MB)")
    return model


def estimate_vram(batch_size=256):
    """Rough VRAM estimate in MB."""
    # Model params: ~2.5M × 4 bytes
    model_mb = sum([
        3 * 32 * 3 * 3 * 4,    # conv1 weights
        32 * 64 * 3 * 3 * 4,   # conv2 weights
        64 * 64 * 3 * 3 * 4,   # conv3 weights
        6400 * 512 * 4,         # fc1
        512 * 256 * 4,          # fc2
        256 * 100 * 4,          # fc3
    ]) / 1e6

    # Activations per batch
    act_mb = batch_size * (
        3 * 10 * 10 +    # input
        32 * 10 * 10 +   # conv1
        64 * 10 * 10 +   # conv2
        64 * 10 * 10 +   # conv3
        512 + 256 + 100  # fc
    ) * 4 / 1e6

    # Gradients ≈ same as params
    total = model_mb * 3 + act_mb
    return total


if __name__ == "__main__":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(device)
    vram   = estimate_vram(config.BATCH_SIZE)
    print(f"[Model] Estimated VRAM: {vram:.0f} MB  (budget: 4000 MB)")

    # Test forward pass
    dummy = torch.randn(4, 3, 10, 10).to(device)
    out   = model(dummy)
    print(f"[Model] Output shape: {out.shape}  ✓")
    print(f"[Model] Output sums to 1: {out[0].sum().item():.4f}")
