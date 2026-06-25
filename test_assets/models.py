from __future__ import annotations

import torch
import torch.nn as nn


class TinyRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 8,
        output_dim: int = 2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = self.net(x)
        return {
            "prediction": prediction,
            "mean_prediction": prediction.mean(dim=-1),
        }

