from __future__ import annotations

from typing import Any

from src.core.agents import AgentConfig
from src.core.session import DebateEvent, DebateSnapshot, EventType
from src.core.stance_tracking import (
    build_recent_positions,
    detect_unresolved_conflicts,
    get_last_speaker_snapshot,
)


def _event_type_label(event: DebateEvent) -> str:
    label = event.event_type.value.upper()
    if event.event_type == EventType.HUMAN_STEER:
        label = "HUMAN_STEER"
    elif event.event_type == EventType.HUMAN_MESSAGE:
        label = "HUMAN_START"
    elif event.event_type == EventType.AGENT_MESSAGE:
        label = "AGENT"
    elif event.event_type == EventType.AGENT_PASS:
        label = "AGENT_PASS"
    elif event.event_type == EventType.AGENT_STALE:
        label = "AGENT_STALE"
    return label


def _event_to_block(event: DebateEvent) -> str:
    label = _event_type_label(event)
    actor = event.actor_name or "Unknown"

    content = event.content.replace("\n", " ").strip()
    if len(content) > 240:
        content = content[:237].rstrip() + "..."

    msg_id = event.metadata.get("discord_message_id")
    reply_to_id = event.metadata.get("reply_to_message_id")
    reply_to_actor = event.metadata.get("reply_to_actor_name")
    reply_to = "none"
    if reply_to_id:
        target = str(reply_to_actor or "unknown")
        reply_to = f"{target}#{reply_to_id}"

    msg_text = str(msg_id) if msg_id else "n/a"
    return (
        f"[EVENT] type={label} turn={event.turn_index} epoch={event.epoch}\n"
        f"speaker={actor}\n"
        f"message_id={msg_text} reply_to={reply_to}\n"
        f"content={content}"
    )


def _build_awareness_snapshot(events: list[DebateEvent]) -> tuple[str, str, str]:
    last = get_last_speaker_snapshot(events)
    if last is None:
        last_speaker = "none"
    else:
        tags = ", ".join(last.tags) if last.tags else "none"
        mid = last.message_id or "n/a"
        last_speaker = (
            f"speaker={last.actor_name} message_id={mid} tags={tags}\n"
            f"point={last.summary}"
        )

    positions = build_recent_positions(events, max_agents=8)
    if not positions:
        positions_text = "none"
    else:
        lines = []
        for actor_name, snapshot in positions.items():
            tags = ", ".join(snapshot.tags) if snapshot.tags else "none"
            lines.append(f"- {actor_name}: tags=[{tags}] point={snapshot.summary}")
        positions_text = "\n".join(lines)

    conflicts = detect_unresolved_conflicts(positions)
    if not conflicts:
        conflicts_text = "none"
    else:
        conflicts_text = "\n".join(f"- {entry}" for entry in conflicts)

    return last_speaker, positions_text, conflicts_text


def _trim_events(events: list[DebateEvent], max_events: int, max_chars: int) -> list[DebateEvent]:
    human_events = [e for e in events if e.event_type in {EventType.HUMAN_MESSAGE, EventType.HUMAN_STEER}]
    non_human = [e for e in events if e.event_type not in {EventType.HUMAN_MESSAGE, EventType.HUMAN_STEER, EventType.SESSION_STARTED}]

    remaining_capacity = max(0, max_events - len(human_events))
    selected_non_human = non_human[-remaining_capacity:] if remaining_capacity else []

    selected_ids = {e.event_id for e in human_events + selected_non_human}
    merged = [e for e in events if e.event_id in selected_ids]

    while True:
        transcript = "\n\n".join(_event_to_block(e) for e in merged)
        if len(transcript) <= max_chars:
            return merged

        # Preserve all human entries, drop oldest non-human first.
        dropped = False
        for idx, candidate in enumerate(merged):
            if candidate.event_type in {EventType.HUMAN_MESSAGE, EventType.HUMAN_STEER}:
                continue
            del merged[idx]
            dropped = True
            break
        if not dropped:
            return merged


def build_agent_messages(
    snapshot: DebateSnapshot,
    agent: AgentConfig,
    *,
    other_agents: list[AgentConfig] | None = None,
    max_events: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    selected_events = _trim_events(list(snapshot.events), max_events=max_events, max_chars=max_chars)
    transcript = "\n\n".join(_event_to_block(event) for event in selected_events)
    last_speaker_text, positions_text, conflicts_text = _build_awareness_snapshot(selected_events)

    participants_info = ""
    if other_agents:
        other_names = ", ".join(a.name for a in other_agents)
        participants_info = f"Other council participants currently present: {other_names}\n"

    system_prompt = (
        f"{agent.system_prompt}\n\n"
        "You are part of a multi-agent council in a Discord group chat.\n"
        f"{participants_info}"
        "All events above are visible to all agents.\n"
        "Human follow-up messages labeled HUMAN_STEER are steering instructions and must be prioritized.\n"
        "Use the awareness snapshot to reference recent positions before proposing new points.\n"
        "CRITICAL: When responding to a specific point made by another council member, ALWAYS set reply_to_message_id to their message_id from the transcript. Do NOT just mention their name in your message text — use the reply feature to keep the conversation threaded.\n"
        "Do NOT send messages like 'I agree with X' or 'I have no additional points' — if you have nothing new to add, use action 'pass' instead.\n"
        "CRITICAL: Keep each message short and natural, like a real Discord chat message. Aim for 50-200 characters per message. Never exceed 400 characters. Use short paragraphs or single lines. Avoid walls of text, numbered lists, or essay-style formatting.\n"
        "You can send multiple messages in sequence by returning multiple JSON objects, one after another. Each JSON object represents one separate message. Use this to reply to multiple people or break up your thoughts naturally.\n"
        "Example of multiple messages:\n"
        '{"action": "speak", "message": "First point here", "reply_to_message_id": "123"}\n'
        '{"action": "speak", "message": "Also, second point"}\n'
        "Do NOT use numbered lists or bullet points in a single message. Instead, send multiple JSON blocks.\n"
        "Do NOT include message_id numbers, debug info, or 'Reply to X' text in your message content. The reply_to_message_id field handles threading automatically.\n"
        "Return strict JSON only with keys: action, message, and optionally reply_to_message_id.\n"
        "- action must be either speak or pass\n"
        "- if action is pass, you can pass if you have fully expressed your viewpoint\n"
        "- if action is speak, message should be concise, natural Discord-style chat text\n"
        "- (Optional) reply_to_message_id: the message_id of the specific message you are responding to (must be a numeric string)\n"
        "Tools (web_search, execute_python, execute_javascript) are available but use them only when they genuinely add value.\n"
        "Do not use tools for simple computations, logging, or statements you already know.\n"
        "Do not speculate about tool costs or availability - they are already configured.\n"
        "IMPORTANT: Your response must be ONLY the JSON object(s). Do NOT include any other text, explanations, or system information.\n"
    )

    user_prompt = (
        f"Session ID: {snapshot.session_id}\n"
        f"Current Turn: {snapshot.turn_index}\n"
        f"Current Epoch: {snapshot.epoch}\n\n"
        "Last speaker snapshot:\n"
        f"{last_speaker_text}\n\n"
        "Recent positions:\n"
        f"{positions_text}\n\n"
        "Potential unresolved conflicts:\n"
        f"{conflicts_text}\n\n"
        "Structured conversation transcript:\n"
        f"{transcript}\n\n"
        "Respond now with your JSON decision."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_moderator_messages(
    snapshot: DebateSnapshot,
    moderator: AgentConfig,
    *,
    force: bool,
    max_chars: int,
) -> list[dict[str, str]]:
    events = list(snapshot.events)
    transcript = "\n\n".join(_event_to_block(event) for event in events)
    _, positions_text, conflicts_text = _build_awareness_snapshot(events)
    if len(transcript) > max_chars:
        transcript = transcript[-max_chars:]

    instruction = (
        "Create a thorough summary and comprehensive overview of the discussion.\n"
        "Highlight what each agent thought, the main arguments presented, and key interactions.\n"
        "Keep it well-structured and end with: DEBATE CLOSED."
    )
    if force:
        instruction = (
            "Hard stop reached. Force close now. "
            "Ensure you synthesize each active viewpoint fairly.\n"
            + instruction
        )

    return [
        {"role": "system", "content": moderator.system_prompt},
        {
            "role": "user",
            "content": (
                f"{instruction}\n\n"
                f"Recent positions:\n{positions_text}\n\n"
                f"Open conflicts:\n{conflicts_text}\n\n"
                f"Transcript:\n{transcript}"
            ),
        },
    ]
