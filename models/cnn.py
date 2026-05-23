"""
Person 2 — CNN
Reads the stacked game frames and encodes them into a 512-dim feature vector.
Architecture from DeepMind's DQN paper (Mnih et al., 2015).

Input:  (batch, 4, 84, 84)  — 4 grayscale frames stacked
Output: (batch, 512)        — compact spatial + temporal encoding
"""

import torch
import torch.nn as nn


class DonkeyKongCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Three conv layers progressively extract spatial features.
        # Stacking 4 frames gives the network motion information (barrel direction, jump arc).
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # (4,84,84) → (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → (64,9,9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → (64,7,7)
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x: (batch, 4, 84, 84) float32 tensor, values in [0, 1]
        returns: (batch, 512) feature vector
        """
        return self.fc(self.conv(x))

    @property
    def output_dim(self):
        return 512


if __name__ == '__main__':
    model = DonkeyKongCNN()
    dummy = torch.zeros(1, 4, 84, 84)
    out = model(dummy)
    print(f"CNN output shape: {out.shape}")  # should be (1, 512)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
