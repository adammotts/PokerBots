from collections import OrderedDict
from typing import Any

import numpy as np
import numpy.typing as npt
from rlcard.games.base import Card


class State:
    def __init__(
        self,
        *,
        obs: npt.NDArray[np.float64],
        legal_actions: OrderedDict[int, None],
        player_id: int,
        hand: tuple[Card, Card],
        **kwargs: dict[str, Any],
    ) -> None:
        self.obs: npt.NDArray[np.float64] = obs
        self.legal_actions: OrderedDict[int, None] = legal_actions
        self.player_id: int = player_id
        self.hand: tuple[Card, Card] = hand
