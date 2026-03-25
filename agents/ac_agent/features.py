import numpy as np
import torch

from env.state import State


def build_features(state: State, device: str = "cpu") -> torch.Tensor:
    """Build 77-dim feature vector from a game State.

    Layout:
      [0:72]  RLCard obs vector (card one-hots + raise count encoding)
      [72:76] Legal action mask (call=0, raise=1, fold=2, check=3)
      [76]    Player position (0=SB, 1=BB)
    """
    obs = torch.from_numpy(state.obs).float()
    legal_mask = torch.zeros(4)
    for a in state.legal_actions:
        legal_mask[a] = 1.0
    position = torch.tensor([float(state.player_id)])
    return torch.cat([obs, legal_mask, position]).to(device)


def build_opponent_summary(
    action_record: list[tuple[int, str]],
    our_player_id: int,
    payoff: float,
    went_to_showdown: bool,
    device: str = "cpu",
) -> torch.Tensor:
    """Build 8-dim summary of opponent behavior in one completed hand.

    Layout:
      [0:4] Normalized opponent action frequencies (call, raise, fold, check)
      [4]   Showdown result (+1 win, -1 loss, 0 no showdown)
      [5]   Went to showdown (binary)
      [6]   Payoff normalized by big blind (payoff / 10)
      [7]   Number of betting rounds reached (normalized 0.25-1.0)
    """
    counts = np.zeros(4, dtype=np.float32)
    action_map = {"call": 0, "raise": 1, "fold": 2, "check": 3}

    for pid, action_str in action_record:
        if pid != our_player_id:
            idx = action_map.get(action_str)
            if idx is not None:
                counts[idx] += 1

    total = counts.sum()
    freqs = counts / max(total, 1.0)

    showdown_result = 0.0
    if went_to_showdown:
        showdown_result = 1.0 if payoff > 0 else -1.0

    num_actions = len(action_record)
    rounds_est = min(1.0 + num_actions / 4.0, 4.0)
    rounds_norm = rounds_est / 4.0

    summary = torch.tensor(
        [
            freqs[0],
            freqs[1],
            freqs[2],
            freqs[3],
            showdown_result,
            float(went_to_showdown),
            payoff / 10.0,
            rounds_norm,
        ],
        dtype=torch.float32,
    )
    return summary.to(device)
