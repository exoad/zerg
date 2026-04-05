from __future__ import annotations

from collections import deque

from src.core.agents import AgentConfig


class WeightedRoundRobinScheduler:
    def __init__(self, agents: list[AgentConfig]) -> None:
        if not agents:
            raise ValueError("At least one debate agent is required")

        self._base_cycle: list[AgentConfig] = []
        for agent in agents:
            self._base_cycle.extend([agent] * max(1, agent.weight))

        self._index = 0
        self._priority_queue: deque[AgentConfig] = deque()

    def next_agent(self) -> AgentConfig:
        if self._priority_queue:
            return self._priority_queue.popleft()

        agent = self._base_cycle[self._index]
        self._index = (self._index + 1) % len(self._base_cycle)
        return agent

    def push_front(self, agent: AgentConfig) -> None:
        self._priority_queue.appendleft(agent)
