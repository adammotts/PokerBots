import torch
import torch.nn as nn


class OpponentLSTM(nn.Module):
    """Cross-hand opponent model.

    Processes a summary of each completed hand as a time series.
    Hidden state persists across hands within a session and encodes
    an implicit model of the opponent's tendencies.
    """

    def __init__(self, input_size: int = 8, hidden_size: int = 32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

    def forward(
        self,
        hand_summary: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # hand_summary: (1, 1, 8)
        # hidden: (h, c) each (1, 1, 32)
        output, hidden_new = self.lstm(hand_summary, hidden)
        return output, hidden_new

    def init_hidden(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, 1, self.hidden_size, device=device),
            torch.zeros(1, 1, self.hidden_size, device=device),
        )


class ActorNetwork(nn.Module):
    """Policy network: state features + opponent context -> action logits.

    Contains a game LSTM that captures within-hand action sequences.
    This LSTM should be reset at the start of each hand.
    """

    def __init__(
        self,
        state_dim: int = 77,
        hidden_dim: int = 64,
        opp_dim: int = 32,
        num_actions: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, hidden_dim)
        self.game_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head1 = nn.Linear(hidden_dim + opp_dim, 32)
        self.head2 = nn.Linear(32, num_actions)

    def forward(
        self,
        features: torch.Tensor,
        game_hidden: tuple[torch.Tensor, torch.Tensor],
        opp_context: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # features: (1, 77)
        # game_hidden: (h, c) each (1, 1, hidden_dim)
        # opp_context: (1, 32)
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        x = x.unsqueeze(1)  # (1, 1, hidden_dim)
        x, game_hidden_new = self.game_lstm(x, game_hidden)
        x = x.squeeze(1)  # (1, hidden_dim)
        combined = torch.cat([x, opp_context], dim=-1)
        x = torch.relu(self.head1(combined))
        logits = self.head2(x)
        return logits, game_hidden_new

    def init_game_hidden(
        self, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, 1, self.hidden_dim, device=device),
            torch.zeros(1, 1, self.hidden_dim, device=device),
        )


class CriticNetwork(nn.Module):
    """Value network: state features + opponent context -> V(s).

    Same structure as ActorNetwork but outputs a scalar state value.
    """

    def __init__(
        self,
        state_dim: int = 77,
        hidden_dim: int = 64,
        opp_dim: int = 32,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, hidden_dim)
        self.game_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head1 = nn.Linear(hidden_dim + opp_dim, 32)
        self.head2 = nn.Linear(32, 1)

    def forward(
        self,
        features: torch.Tensor,
        game_hidden: tuple[torch.Tensor, torch.Tensor],
        opp_context: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        x = x.unsqueeze(1)
        x, game_hidden_new = self.game_lstm(x, game_hidden)
        x = x.squeeze(1)
        combined = torch.cat([x, opp_context], dim=-1)
        x = torch.relu(self.head1(combined))
        value = self.head2(x)
        return value, game_hidden_new

    def init_game_hidden(
        self, device: str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(1, 1, self.hidden_dim, device=device),
            torch.zeros(1, 1, self.hidden_dim, device=device),
        )


class ConfidenceGate(nn.Module):
    """Learned confidence signal from the opponent LSTM hidden state.

    Outputs a scalar in [0, 1] indicating how confident the opponent
    model is. Used to gate KL regularization: high confidence -> low
    KL penalty -> more exploitation.
    """

    def __init__(self, opp_hidden_size: int = 32) -> None:
        super().__init__()
        self.linear = nn.Linear(opp_hidden_size, 1)

    def forward(self, opp_hidden_h: torch.Tensor) -> torch.Tensor:
        # opp_hidden_h: (1, 1, 32) -> squeeze to (1, 32)
        h = opp_hidden_h.squeeze(0)
        return torch.sigmoid(self.linear(h))
