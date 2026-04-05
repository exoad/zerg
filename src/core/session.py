from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    SESSION_STARTED = "session_started"
    HUMAN_MESSAGE = "human_message"
    HUMAN_STEER = "human_steer"
    AGENT_MESSAGE = "agent_message"
    AGENT_PASS = "agent_pass"
    AGENT_STALE = "agent_stale"
    CONSENSUS_REACHED = "consensus_reached"
    FORCED_FINALIZE = "forced_finalize"
    SESSION_CLOSED = "session_closed"


@dataclass(frozen=True)
class DebateEvent:
    event_id: str
    parent_event_id: str | None
    event_type: EventType
    created_at: datetime
    epoch: int
    turn_index: int
    actor_id: str | None
    actor_name: str | None
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DebateSnapshot:
    session_id: str
    channel_id: int
    starter_user_id: int
    events: tuple[DebateEvent, ...]
    turn_index: int
    epoch: int
    active: bool
    created_at: datetime


@dataclass
class DebateSession:
    session_id: str
    channel_id: int
    starter_user_id: int
    max_turns: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    events: list[DebateEvent] = field(default_factory=list)
    turn_index: int = 0
    epoch: int = 0
    active: bool = True

    inflight_agent: str | None = None
    inflight_epoch: int | None = None
    inflight_task: asyncio.Task[Any] | None = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)

    def _append_event(
        self,
        *,
        event_type: EventType,
        actor_id: str | None,
        actor_name: str | None,
        content: str,
        metadata: dict[str, Any] | None = None,
        increment_turn: bool,
    ) -> DebateEvent:
        if increment_turn:
            self.turn_index += 1

        parent_event_id = self.events[-1].event_id if self.events else None
        event = DebateEvent(
            event_id=str(uuid.uuid4()),
            parent_event_id=parent_event_id,
            event_type=event_type,
            created_at=datetime.now(timezone.utc),
            epoch=self.epoch,
            turn_index=self.turn_index,
            actor_id=actor_id,
            actor_name=actor_name,
            content=content,
            metadata=metadata or {},
        )
        self.events.append(event)
        return event

    async def start(self, opener_name: str, opener_text: str) -> DebateEvent:
        async with self.lock:
            if self.events:
                raise ValueError("Session already started")

            self._append_event(
                event_type=EventType.SESSION_STARTED,
                actor_id=str(self.starter_user_id),
                actor_name=opener_name,
                content="Council session started",
                metadata={},
                increment_turn=False,
            )
            return self._append_event(
                event_type=EventType.HUMAN_MESSAGE,
                actor_id=str(self.starter_user_id),
                actor_name=opener_name,
                content=opener_text.strip(),
                metadata={"human_role": "starter"},
                increment_turn=False,
            )

    async def add_human_input(self, user_id: int, display_name: str, content: str) -> DebateEvent:
        async with self.lock:
            if not self.active:
                raise ValueError("Session is already closed")

            has_starter_message = any(e.event_type == EventType.HUMAN_MESSAGE for e in self.events)
            event_type = EventType.HUMAN_STEER if has_starter_message else EventType.HUMAN_MESSAGE

            if event_type == EventType.HUMAN_STEER:
                self.epoch += 1
                self.interrupt_event.set()

            return self._append_event(
                event_type=event_type,
                actor_id=str(user_id),
                actor_name=display_name,
                content=content.strip(),
                metadata={"human_role": "steer" if event_type == EventType.HUMAN_STEER else "starter"},
                increment_turn=False,
            )

    async def set_inflight(self, agent_name: str, task: asyncio.Task[Any]) -> int:
        async with self.lock:
            self.inflight_agent = agent_name
            self.inflight_epoch = self.epoch
            self.inflight_task = task
            return self.epoch

    async def clear_inflight(self) -> None:
        async with self.lock:
            self.inflight_agent = None
            self.inflight_epoch = None
            self.inflight_task = None

    async def pop_interrupt(self) -> bool:
        async with self.lock:
            interrupted = self.interrupt_event.is_set()
            self.interrupt_event.clear()
            return interrupted

    async def add_agent_message(
        self,
        agent_name: str,
        content: str,
        *,
        stale: bool,
        metadata: dict[str, Any] | None = None,
    ) -> DebateEvent:
        async with self.lock:
            event_type = EventType.AGENT_STALE if stale else EventType.AGENT_MESSAGE
            event_metadata = {"stale": stale}
            if metadata:
                event_metadata.update(metadata)
            return self._append_event(
                event_type=event_type,
                actor_id=None,
                actor_name=agent_name,
                content=content.strip(),
                metadata=event_metadata,
                increment_turn=not stale,
            )

    async def add_agent_pass(
        self,
        agent_name: str,
        reason: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> DebateEvent:
        async with self.lock:
            return self._append_event(
                event_type=EventType.AGENT_PASS,
                actor_id=None,
                actor_name=agent_name,
                content=reason.strip() or "pass",
                metadata=metadata or {},
                increment_turn=True,
            )

    async def mark_consensus(self, summary: str) -> DebateEvent:
        async with self.lock:
            self.active = False
            return self._append_event(
                event_type=EventType.CONSENSUS_REACHED,
                actor_id=None,
                actor_name="Moderator",
                content=summary.strip(),
                metadata={},
                increment_turn=False,
            )

    async def mark_forced_finalize(self, summary: str) -> DebateEvent:
        async with self.lock:
            self.active = False
            return self._append_event(
                event_type=EventType.FORCED_FINALIZE,
                actor_id=None,
                actor_name="Moderator",
                content=summary.strip(),
                metadata={},
                increment_turn=False,
            )

    async def close(self, reason: str) -> DebateEvent:
        async with self.lock:
            self.active = False
            return self._append_event(
                event_type=EventType.SESSION_CLOSED,
                actor_id=None,
                actor_name="System",
                content=reason.strip(),
                metadata={},
                increment_turn=False,
            )

    async def should_force_finalize(self) -> bool:
        async with self.lock:
            return self.turn_index >= self.max_turns

    async def snapshot(self) -> DebateSnapshot:
        async with self.lock:
            return DebateSnapshot(
                session_id=self.session_id,
                channel_id=self.channel_id,
                starter_user_id=self.starter_user_id,
                events=tuple(self.events),
                turn_index=self.turn_index,
                epoch=self.epoch,
                active=self.active,
                created_at=self.created_at,
            )
