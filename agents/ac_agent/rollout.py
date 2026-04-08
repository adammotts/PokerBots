from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class StepData:
    features: torch.Tensor
    action: int
    log_prob_old: float
    legal_mask: torch.Tensor


@dataclass
class HandRollout:
    steps: list[StepData] = field(default_factory=list)
    reward: float = 0.0
    hand_summary: torch.Tensor = field(default_factory=lambda: torch.zeros(8))
    opp_freq_target: torch.Tensor = field(default_factory=lambda: torch.zeros(4))
