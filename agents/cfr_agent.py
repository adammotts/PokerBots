import shutil
import tempfile
from collections import OrderedDict

import numpy as np
import numpy.typing as npt
import rlcard
from rlcard.agents.cfr_agent import CFRAgent as RLCardCFRAgent

from agents.base_agent import BaseAgent, Transition


class CFRAgent(BaseAgent):
    """Wraps RLCard's CFR agent behind the BaseAgent interface.

    Used as a baseline: trains via game-tree traversal, then acts
    according to the learned average strategy at eval time.
    """

    def __init__(
        self,
        model_path: str = "./cfr_model",
        iterations: int = 1000,
    ) -> None:
        self.iterations = iterations
        self._env: rlcard.envs.Env = rlcard.make(
            "limit-holdem",
            {"allow_step_back": True},
        )
        self._cfr = RLCardCFRAgent(self._env, model_path)

    def act(
        self,
        obs: npt.NDArray[np.float64],
        legal_actions: list[int],
        *,
        training: bool = True,
        raw_obs: dict[str, object] | None = None,
        action_record: list[tuple[int, str]] | None = None,
        player_id: int = 0,
    ) -> int:
        state: dict[str, object] = {
            "obs": obs,
            "legal_actions": OrderedDict.fromkeys(legal_actions),
        }
        action, _ = self._cfr.eval_step(state)
        return int(action)

    def observe(self, transition: Transition) -> None:
        pass

    def update(self) -> None:
        for _ in range(self.iterations):
            self._cfr.train()

    def save(self, path: str) -> None:
        # Write to a temp dir on the same filesystem, then rename each file
        # atomically so a killed process never leaves a partial checkpoint.
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(path, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp_dir:
            self._cfr.model_path = tmp_dir
            self._cfr.save()
            for fname in os.listdir(tmp_dir):
                os.replace(
                    os.path.join(tmp_dir, fname),
                    os.path.join(path, fname),
                )
        self._cfr.model_path = path

    def load(self, path: str) -> None:
        self._cfr.model_path = path
        self._cfr.load()
