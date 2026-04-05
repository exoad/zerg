from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.session import DebateEvent, EventType


_SENTENCE_RE = re.compile(r"[.!?]\s+")

# Lightweight keyword map for deterministic, low-cost stance extraction.
_STANCE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "support": ("agree", "support", "endorse", "favorable", "yes"),
    "oppose": ("disagree", "oppose", "reject", "avoid", "no ", "not viable"),
    "caution": ("risk", "concern", "caution", "trade-off", "downside", "uncertain"),
    "speed": ("fast", "quick", "immediate", "asap", "rapid"),
    "robustness": ("reliable", "robust", "safe", "testing", "resilient", "stable"),
    "cost": ("cost", "budget", "cheap", "free-tier", "efficient"),
    "scale": ("scale", "scalable", "long-term", "maintain", "sustainable"),
}


@dataclass(frozen=True)
class PositionSnapshot:
    actor_name: str
    summary: str
    tags: tuple[str, ...]
    message_id: str | None


def summarize_position(text: str, *, max_chars: int = 180) -> str:
    compact = " ".join(text.replace("\n", " ").split()).strip()
    if not compact:
        return "(no content)"

    first = _SENTENCE_RE.split(compact, maxsplit=1)[0].strip()
    if not first:
        first = compact

    if len(first) <= max_chars:
        return first
    return first[: max_chars - 3].rstrip() + "..."


def extract_stance_tags(text: str) -> set[str]:
    lowered = text.lower()
    tags: set[str] = set()
    for tag, keywords in _STANCE_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            tags.add(tag)
    return tags


def _is_active_agent_message(event: DebateEvent) -> bool:
    return event.event_type == EventType.AGENT_MESSAGE and not event.metadata.get("stale", False)


def _to_snapshot(event: DebateEvent) -> PositionSnapshot:
    actor = event.actor_name or "Unknown"
    return PositionSnapshot(
        actor_name=actor,
        summary=summarize_position(event.content),
        tags=tuple(sorted(extract_stance_tags(event.content))),
        message_id=(str(event.metadata.get("discord_message_id")) if event.metadata.get("discord_message_id") else None),
    )


def get_last_speaker_snapshot(events: list[DebateEvent]) -> PositionSnapshot | None:
    for event in reversed(events):
        if _is_active_agent_message(event):
            return _to_snapshot(event)
    return None


def build_recent_positions(events: list[DebateEvent], *, max_agents: int = 6) -> dict[str, PositionSnapshot]:
    by_agent: dict[str, PositionSnapshot] = {}
    for event in reversed(events):
        if not _is_active_agent_message(event):
            continue
        if not event.actor_name:
            continue
        if event.actor_name in by_agent:
            continue

        by_agent[event.actor_name] = _to_snapshot(event)
        if len(by_agent) >= max(1, max_agents):
            break

    return dict(sorted(by_agent.items(), key=lambda item: item[0].lower()))


def detect_unresolved_conflicts(positions: dict[str, PositionSnapshot]) -> list[str]:
    if len(positions) < 2:
        return []

    tagged: dict[str, set[str]] = {name: set(snapshot.tags) for name, snapshot in positions.items()}

    support_agents = sorted(name for name, tags in tagged.items() if "support" in tags)
    oppose_agents = sorted(name for name, tags in tagged.items() if "oppose" in tags)
    speed_agents = sorted(name for name, tags in tagged.items() if "speed" in tags)
    caution_agents = sorted(name for name, tags in tagged.items() if "caution" in tags)
    robustness_agents = sorted(name for name, tags in tagged.items() if "robustness" in tags)

    conflicts: list[str] = []
    if support_agents and oppose_agents:
        conflicts.append(
            "support-vs-oppose: "
            f"support={', '.join(support_agents)} | oppose={', '.join(oppose_agents)}"
        )

    if speed_agents and caution_agents:
        conflicts.append(
            "speed-vs-caution: "
            f"speed={', '.join(speed_agents)} | caution={', '.join(caution_agents)}"
        )

    if speed_agents and robustness_agents:
        conflicts.append(
            "speed-vs-robustness: "
            f"speed={', '.join(speed_agents)} | robustness={', '.join(robustness_agents)}"
        )

    return conflicts
