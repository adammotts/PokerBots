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
    if spec.agent_type == "ac":
        agent = ActorCriticAgent()
        agent.load(str(models_dir / spec.model_name / "final.pt"))
        return ActorCriticPlayer(agent=agent)

    if spec.agent_type == "dqn":
        agent = DoubleDQNAgent()
        agent.load(str(models_dir / spec.model_name / "final.pt"))
        return DoubleDQNPlayer(agent=agent)

    raise ValueError(f"Unsupported agent type: {spec.agent_type}")
