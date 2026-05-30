"""
Person 2 — CNN feature extractor.
Architecture follows the DeepMind DQN Nature paper (2015), adapted for NES.
Input:  (batch, 4, 84, 84)  — 4 stacked grayscale frames
Output: (batch, num_actions) — Q-value per action
"""

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    Three convolutional layers that compress a stack of frames into a
    512-dimensional feature vector.

    Conv output size derivation for 84x84 input:
        Conv1  kernel=8 stride=4  → 20x20
        Conv2  kernel=4 stride=2  →  9x9
        Conv3  kernel=3 stride=1  →  7x7
        Flatten: 64 * 7 * 7 = 3136
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    @staticmethod
    def feature_dim() -> int:
        return 512


class DQNNetwork(nn.Module):
    """
    Full DQN: CNN feature extractor + linear head → Q-values.
    """

    def __init__(self, num_actions: int, in_channels: int = 4):
        super().__init__()
        self.features = CNNFeatureExtractor(in_channels)
        self.head = nn.Linear(CNNFeatureExtractor.feature_dim(), num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
