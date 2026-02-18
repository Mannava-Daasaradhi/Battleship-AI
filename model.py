# ============================================================
#  model.py
#  CNN policy network for the RL attacker.
#  Sized to fit comfortably in 4GB VRAM (GTX 1650).
#
#  Input : (B, 3, 10, 10) — hit/miss/unknown board states
#  Output: (B, 100)       — shot probability per cell
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class BattleshipNet(nn.Module):
    def __init__(self):
        super().__init__()

        ch = config.RL_CONV_CHANNELS   # [32, 64, 64]

        self.conv = nn.Sequential(
            nn.Conv2d(config.RL_INPUT_CHANNELS, ch[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch[0], ch[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch[1], ch[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(inplace=True),
        )

        fc_in = ch[2] * config.BOARD_SIZE * config.BOARD_SIZE  # 64*10*10 = 6400
        fc    = config.RL_FC_DIMS                               # [512, 256]

        self.fc = nn.Sequential(
            nn.Linear(fc_in, fc[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(fc[0], fc[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc[1], config.BOARD_SIZE ** 2),           # 100 logits
        )

    def forward(self, x):
        """Returns (B, 100) softmax probabilities."""
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return F.softmax(self.fc(h), dim=-1)

    def logits(self, x):
        """Returns (B, 100) raw logits (for loss computation)."""
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


def build_model(device):
    model  = BattleshipNet().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {params:,} parameters  (~{params*4/1e6:.1f} MB)")
    return model


def load_model(path, device):
    """Load model from checkpoint. Returns model in eval mode."""
    model = build_model(device)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


if __name__ == "__main__":
    import torch
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(dev)
    x     = torch.randn(4, 3, 10, 10).to(dev)
    out   = model(x)
    print(f"Output shape : {out.shape}")
    print(f"Output sum   : {out[0].sum().item():.4f}  (should be 1.0)")
