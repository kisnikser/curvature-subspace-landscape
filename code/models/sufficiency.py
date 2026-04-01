"""LSTM heads for criterion-specific sample-sufficiency prediction."""

from __future__ import annotations

import torch
from torch import nn


class CriterionSpecificLSTM(nn.Module):
    """Predict sufficiency probability and sufficient sample size from a short history."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output, _state = self.lstm(x)
        last_hidden = self.norm(output[:, -1, :])
        return {
            "logits": self.classifier(last_hidden).squeeze(-1),
            "log_kstar": self.regressor(last_hidden).squeeze(-1),
        }
