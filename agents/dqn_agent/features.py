from __future__ import annotations

import torch

from env.hand_strength import evaluate_hand_strength
from env.state import State

RL_OBS_DIM = 72
HAND_RANK_DIM = 10
DRAW_FLAG_DIM = 3
LEGAL_ACTION_DIM = 4
POSITION_DIM = 1

HAND_FEATURE_START = RL_OBS_DIM
LEGAL_ACTION_START = HAND_FEATURE_START + HAND_RANK_DIM + DRAW_FLAG_DIM
POSITION_START = LEGAL_ACTION_START + LEGAL_ACTION_DIM
STATE_DIM = POSITION_START + POSITION_DIM


def build_dqn_features(state: State, device: str = "cpu") -> torch.Tensor:
    """Build DQN features with engineered hand-strength signals.

    Layout:
      [0:72]   RLCard obs vector
      [72:82]  Hand rank one-hot (1-10)
      [82]     Flush draw flag
      [83]     Straight draw flag
      [84]     Boat draw flag
      [85:89]  Legal action mask (call=0, raise=1, fold=2, check=3)
      [89]     Player position (0=SB, 1=BB)
    """
    obs = torch.from_numpy(state.obs).float()

    strength = evaluate_hand_strength(state.hand, state.board)
    hand_rank = torch.zeros(HAND_RANK_DIM)
    hand_rank[strength.hand_rank - 1] = 1.0
    draw_flags = torch.tensor(
        [
            float(strength.has_flush_draw),
            float(strength.has_straight_draw),
            float(strength.has_boat_draw),
        ]
    )

    legal_mask = torch.zeros(LEGAL_ACTION_DIM)
    for action in state.legal_actions:
        legal_mask[action] = 1.0

    position = torch.tensor([float(state.player_id)])
    return torch.cat([obs, hand_rank, draw_flags, legal_mask, position]).to(device)
