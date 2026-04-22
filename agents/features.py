from __future__ import annotations

import json
import os
from collections.abc import Sequence

import rlcard
import torch
from rlcard.games.base import Card

from env.state import State

CARD_TO_INDEX: dict[str, int] | None = None


def get_rlcard_card_to_index_mapping() -> dict[str, int]:
    global CARD_TO_INDEX
    if CARD_TO_INDEX is None:
        path = os.path.join(rlcard.__path__[0], "games/limitholdem/card2index.json")
        with open(path) as f:
            CARD_TO_INDEX = json.load(f)
    return CARD_TO_INDEX


def build_features(state: State, device: str = "cpu") -> torch.Tensor:
    obs = torch.from_numpy(state.obs).float()
    legal_mask = torch.zeros(4)
    for action in state.legal_actions:
        legal_mask[action] = 1.0
    position = torch.tensor([float(state.player_id)])
    return torch.cat([obs, legal_mask, position]).to(device)


def encode_both_hands_onehot(
    hand_0: Sequence[Card],
    hand_1: Sequence[Card],
    device: str = "cpu",
) -> torch.Tensor:
    c2i = get_rlcard_card_to_index_mapping()
    vec = torch.zeros(104, device=device)
    for card in hand_0:
        key = card.suit + card.rank
        vec[c2i[key]] = 1.0
    for card in hand_1:
        key = card.suit + card.rank
        vec[52 + c2i[key]] = 1.0
    return vec
