from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AgentConfig:
    name: str
    provider: str
    model: str
    avatar_url: str | None
    system_prompt: str
    weight: int = 1
    role: str = "agent"


@dataclass(frozen=True)
class AgentRegistry:
    agents: list[AgentConfig]
    moderator: AgentConfig


def load_agents(path: str | Path) -> AgentRegistry:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_agents = payload.get("agents", [])

    agents: list[AgentConfig] = []
    moderator: AgentConfig | None = None
    seen_names: set[str] = set()

    for raw in raw_agents:
        name = str(raw["name"]).strip()
        role_raw = raw.get("role")
        if role_raw is None:
            role = "moderator" if "moderator" in name.lower() else "agent"
        else:
            role = str(role_raw).strip().lower()
        if role not in {"agent", "moderator"}:
            raise ValueError(f"Invalid role for agent {name}: {role}")

        if name.lower() in seen_names:
            raise ValueError(f"Duplicate agent name in agents.yaml: {name}")
        seen_names.add(name.lower())

        model_raw = raw.get("model")
        model = str(model_raw).strip() if model_raw is not None else ""

        agent = AgentConfig(
            name=name,
            provider=str(raw["provider"]).lower(),
            model=model,
            avatar_url=(str(raw.get("avatar_url", "")).strip() or None),
            system_prompt=str(raw["system_prompt"]),
            weight=max(1, int(raw.get("weight", 1))),
            role=role,
        )
        if agent.role == "moderator":
            if moderator is not None:
                raise ValueError("Multiple moderator profiles found in agents.yaml")
            moderator = agent
            continue
        agents.append(agent)

    if not agents:
        raise ValueError("No active debate agents configured in agents.yaml")

    if len(agents) < 5:
        raise ValueError("At least 5 active debate agents are required in agents.yaml")

    if moderator is None:
        moderator = AgentConfig(
            name="Moderator",
            provider=agents[0].provider,
            model=agents[0].model,
            avatar_url=None,
            system_prompt=(
                "You summarize the council outcome, consensus points,"
                " unresolved disagreements, and next actions."
            ),
            weight=1,
            role="moderator",
        )

    return AgentRegistry(agents=agents, moderator=moderator)
