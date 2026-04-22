from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class StepExperience:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class EpisodeReplayBuffer:
    """Replay buffer that stores complete hands for recurrent training."""

    def __init__(self, capacity: int) -> None:
        self.episodes: deque[list[StepExperience]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.episodes)

    def add_episode(self, episode: list[StepExperience]) -> None:
        if episode:
            self.episodes.append(episode)

    def sample(self, batch_size: int) -> list[list[StepExperience]]:
        return random.sample(self.episodes, k=min(batch_size, len(self.episodes)))
