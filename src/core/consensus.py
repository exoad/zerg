from __future__ import annotations

import re
from itertools import combinations

from src.core.session import DebateEvent, EventType
from src.core.stance_tracking import build_recent_positions, detect_unresolved_conflicts, extract_stance_tags


_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _average_pairwise_jaccard(sets: list[set[str]]) -> float:
    if len(sets) < 2:
        return 0.0
    scores = [_jaccard(a, b) for a, b in combinations(sets, 2)]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def detect_consensus(
    events: list[DebateEvent],
    threshold: float,
    *,
    min_recent_messages: int,
    min_unique_agents: int,
    min_turns: int,
    pass_count_required: int,
    lexical_weight: float = 1.0,
    stance_weight: float = 0.0,
    freshness_weight: float = 0.0,
    disagreement_guard_enabled: bool = False,
) -> bool:
    if not events:
        return False

    latest_turn = max(event.turn_index for event in events)
    if latest_turn < max(0, min_turns):
        return False

    all_agent_msgs = [
        event
        for event in events
        if event.event_type == EventType.AGENT_MESSAGE and not event.metadata.get("stale", False)
    ]
    unique_agents = {event.actor_name for event in all_agent_msgs if event.actor_name}
    if len(unique_agents) < max(1, min_unique_agents):
        return False

    recent_msgs = [
        event
        for event in events
        if event.event_type == EventType.AGENT_MESSAGE and not event.metadata.get("stale", False)
    ][-max(2, min_recent_messages):]

    if len(recent_msgs) < max(2, min_recent_messages):
        return False

    lexical_similarity = _average_pairwise_jaccard([_tokens(event.content) for event in recent_msgs])

    stance_sets = [extract_stance_tags(event.content) for event in recent_msgs]
    stance_convergence = _average_pairwise_jaccard(stance_sets)

    unique_recent_agents = {event.actor_name for event in recent_msgs if event.actor_name}
    freshness = min(1.0, len(unique_recent_agents) / max(1, min_unique_agents))

    weights = [
        max(0.0, lexical_weight),
        max(0.0, stance_weight),
        max(0.0, freshness_weight),
    ]
    weight_total = sum(weights)
    if weight_total <= 0:
        composite = lexical_similarity
    else:
        composite = (
            (weights[0] * lexical_similarity)
            + (weights[1] * stance_convergence)
            + (weights[2] * freshness)
        ) / weight_total

    # If there is a fresh human steer after the first of these messages, avoid early close.
    steer_after = any(
        e.event_type == EventType.HUMAN_STEER and e.created_at > recent_msgs[0].created_at
        for e in events
    )
    if steer_after:
        return False

    if disagreement_guard_enabled:
        positions = build_recent_positions(recent_msgs, max_agents=max(2, min_unique_agents))
        unresolved = detect_unresolved_conflicts(positions)
        if unresolved:
            return False

    if composite >= threshold:
        return True

    required_passes = max(0, pass_count_required)
    if required_passes > 0:
        recent_passes = [event for event in events if event.event_type == EventType.AGENT_PASS][-required_passes:]
        if len(recent_passes) >= required_passes:
            unique_pass_agents = {event.actor_name for event in recent_passes if event.actor_name}
            if len(unique_pass_agents) >= required_passes:
                return True

    return False
