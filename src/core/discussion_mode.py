from __future__ import annotations

import re
from typing import Iterable

from src.core.session import DebateEvent, EventType


DiscussionMode = str
ObjectiveTier = str


def _normalize_mode(value: str, default: str) -> DiscussionMode:
    cleaned = (value or "").strip().lower()
    if cleaned in {"objective", "debatable", "adversarial", "exploratory"}:
        return cleaned
    return default


def _latest_human_text(events: Iterable[DebateEvent]) -> str:
    for event in reversed(list(events)):
        if event.event_type in {EventType.HUMAN_STEER, EventType.HUMAN_MESSAGE}:
            return event.content.strip()
    return ""


def infer_discussion_mode(
    *,
    topic: str,
    events: Iterable[DebateEvent],
    auto_enabled: bool,
    default_mode: str,
) -> DiscussionMode:
    default = _normalize_mode(default_mode, "debatable")
    if not auto_enabled:
        return default

    latest_human = _latest_human_text(events)
    source = f"{topic} {latest_human}".strip().lower()

    # Explicit steer intent should win.
    adversarial_cues = (
        "go savage",
        "be aggressive",
        "attack each other",
        "roast",
        "tear down",
        "no good faith",
        "fight",
        "hard debate",
    )
    if any(cue in source for cue in adversarial_cues):
        return "adversarial"

    exploratory_cues = (
        "texture",
        "color",
        "processing style",
        "compared to other models",
        "describe your own",
        "what does it feel like",
        "if you had to describe",
        "metaphor",
        "vibe",
    )
    if any(cue in source for cue in exploratory_cues):
        return "exploratory"

    objective_patterns = (
        r"\bwhat is\b",
        r"\bdefine\b",
        r"\bexplain\b",
        r"\bhow does\b",
        r"\bhow do\b",
        r"\bscientific method\b",
        r"\bwhen did\b",
        r"\bwho is\b",
        r"\bobjective\b",
        r"\bfactual\b",
    )
    normative_patterns = (
        r"\bshould\b",
        r"\bbetter\b",
        r"\bbest\b",
        r"\bvs\b",
        r"\bversus\b",
        r"\bright or wrong\b",
        r"\bmoral\b",
        r"\bethic\w*\b",
        r"\btrade-?off\b",
    )

    has_objective_signal = any(re.search(pattern, source) for pattern in objective_patterns)
    has_normative_signal = any(re.search(pattern, source) for pattern in normative_patterns)

    if has_objective_signal and not has_normative_signal:
        return "objective"

    if has_normative_signal:
        return "debatable"

    # Question-style topics with low polarity are usually objective.
    if "?" in source and not has_normative_signal:
        return "objective"

    return default


def infer_objective_tier(*, topic: str, events: Iterable[DebateEvent]) -> ObjectiveTier:
    latest_human = _latest_human_text(events)
    source = f"{topic} {latest_human}".strip().lower()

    # Very short, single-answer style objective prompts.
    if re.search(r"\b\d+\s*[+\-*/]\s*\d+\b", source):
        return "simple"

    if re.search(r"\bwhat is\b", source):
        # Tiny "what is X" prompts are usually one-line objective answers.
        token_count = len(re.findall(r"\w+", source))
        if token_count <= 6:
            return "simple"

    simple_markers = (
        "capital of",
        "2+2",
        "1+1",
        "true or false",
    )
    if any(marker in source for marker in simple_markers):
        return "simple"

    if infer_discussion_mode(
        topic=topic,
        events=events,
        auto_enabled=True,
        default_mode="debatable",
    ) == "objective":
        return "explainer"

    return "none"
