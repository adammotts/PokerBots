from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents.ac_agent import ActorCriticAgent
from agents.dqn_agent import DoubleDQNAgent
from players.ac_player import ActorCriticPlayer
from players.base_player import BasePlayer
from players.dqn_player import DoubleDQNPlayer


@dataclass(frozen=True)
class AgentSpec:
    agent_type: str
    model_name: str
    label: str


def parse_agent_spec(spec: str) -> AgentSpec:
    if ":" in spec:
        agent_type, model_name = spec.split(":", maxsplit=1)
        label = spec
    else:
        agent_type = "ac"
        model_name = spec
        label = spec
    return AgentSpec(agent_type=agent_type, model_name=model_name, label=label)


def load_player(spec: AgentSpec, models_dir: Path) -> BasePlayer:
    dir_name = spec.model_name.replace("-", "_")

    if spec.agent_type == "ac":
        agent = ActorCriticAgent()
        ckpt = models_dir / dir_name / "final.pt"
        print(f"Loading AC agent from {ckpt}")
        agent.load(str(ckpt))
        return ActorCriticPlayer(agent=agent)

    if spec.agent_type == "dqn":
        agent = DoubleDQNAgent()
        ckpt = models_dir / dir_name / "final.pt"
        print(f"Loading DQN agent from {ckpt}")
        agent.load(str(ckpt))
        return DoubleDQNPlayer(agent=agent)

    raise ValueError(f"Unsupported agent type: {spec.agent_type}")
