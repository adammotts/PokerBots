from __future__ import annotations

import torch

from env.state import State


def build_features(state: State, device: str = "cpu") -> torch.Tensor:
    """Build a 77-dim feature vector from a game State.

    Layout:
      [0:72]  RLCard obs vector
      [72:76] Legal action mask (call=0, raise=1, fold=2, check=3)
      [76]    Player position (0=SB, 1=BB)
    """
    obs = torch.from_numpy(state.obs).float()
    legal_mask = torch.zeros(4)
    for action in state.legal_actions:
        legal_mask[action] = 1.0
    position = torch.tensor([float(state.player_id)])
    return torch.cat([obs, legal_mask, position]).to(device)
