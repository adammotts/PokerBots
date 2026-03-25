from typing import TypedDict

import numpy as np
import numpy.typing as npt


class ProcessedState(TypedDict):
    obs: npt.NDArray[np.float64]
    legal_actions: list[int]
