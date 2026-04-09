import torch
import torch.nn as nn


class OpponentLSTM(nn.Module):
    def __init__(self, input_size: int = 8, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

    def forward(
        self,
        hand_summary: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        output, hidden_new = self.lstm(hand_summary, hidden)
        return output, hidden_new

    def init_hidden(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, 1, self.hidden_size, device=device),
            torch.zeros(1, 1, self.hidden_size, device=device),
        )


class ActorNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int = 77,
        hidden_dim: int = 128,
        opp_dim: int = 32,
        num_actions: int = 4,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.film_scale = nn.Linear(opp_dim, hidden_dim)
        self.film_shift = nn.Linear(opp_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(
        self,
        features: torch.Tensor,
        opp_context: torch.Tensor,
    ) -> torch.Tensor:
        x = self.trunk(features)
        gamma = self.film_scale(opp_context)
        beta = self.film_shift(opp_context)
        x = gamma * x + beta
        return self.head(x)


class CentralizedCritic(nn.Module):
    def __init__(
        self,
        state_dim: int = 77,
        card_dim: int = 104,
        hidden_dim: int = 128,
        opp_dim: int = 32,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + card_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.film_scale = nn.Linear(opp_dim, hidden_dim)
        self.film_shift = nn.Linear(opp_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        both_hands: torch.Tensor,
        opp_context: torch.Tensor,
    ) -> torch.Tensor:
        x = self.trunk(torch.cat([features, both_hands], dim=-1))
        gamma = self.film_scale(opp_context)
        beta = self.film_shift(opp_context)
        x = gamma * x + beta
        return self.head(x)


class OpponentPredictor(nn.Module):
    def __init__(self, opp_hidden_size: int = 32, num_actions: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(opp_hidden_size, 32)
        self.fc2 = nn.Linear(32, num_actions)

    def forward(self, opp_hidden_h: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(opp_hidden_h))
        return self.fc2(x)
