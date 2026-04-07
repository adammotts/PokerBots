from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Recurrent Q-network for partially observed limit hold'em."""

    def __init__(
        self,
        *,
        state_dim: int = 77,
        hidden_dim: int = 128,
        num_actions: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_actions)

    def forward(
        self,
        features: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        x = x.unsqueeze(1)
        x, hidden_new = self.lstm(x, hidden)
        q_values = self.head(x.squeeze(1))
        return q_values, hidden_new

    def init_hidden(
        self,
        device: str = "cpu",
        batch_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
        )
