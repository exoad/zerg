from __future__ import annotations

from typing import Any

from src.core.agents import AgentConfig
from src.core.discussion_mode import infer_discussion_mode, infer_objective_tier
from src.core.session import DebateEvent, DebateSnapshot, EventType
from src.core.stance_tracking import (
    build_recent_positions,
    detect_unresolved_conflicts,
    get_last_speaker_snapshot,
)


def _safe_intensity(value: str, default: str) -> str:
    cleaned = (value or "").strip().lower()
    if cleaned in {"low", "medium", "high"}:
        return cleaned
    return default


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
    if len(content) > 400:
        content = content[:397].rstrip() + "..."

    msg_id = event.metadata.get("discord_message_id")
    msg_text = str(msg_id) if msg_id else "n/a"
    return (
        f"[EVENT] type={label} turn={event.turn_index} epoch={event.epoch}\n"
        f"speaker={actor}\n"
        f"message_id={msg_text}\n"
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


def _mode_word_band(settings: Any, mode: str, objective_tier: str) -> tuple[int, int]:
    if mode == "objective":
        if objective_tier == "simple":
            return settings.objective_simple_min_words, settings.objective_simple_max_words
        return settings.objective_min_words, settings.objective_max_words
    if mode == "exploratory":
        return settings.debatable_min_words, settings.debatable_max_words
    if mode == "adversarial":
        return settings.adversarial_min_words, settings.adversarial_max_words
    return settings.debatable_min_words, settings.debatable_max_words


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
    topic = snapshot.topic.strip() or "No topic provided"
    if len(topic) > 280:
        topic = topic[:277].rstrip() + "..."

    participants_info = ""
    if other_agents:
        other_names = ", ".join(a.name for a in other_agents)
        participants_info = f"Other participants in this chat: {other_names}\n"

    starter_name = snapshot.starter_display_name.strip() or "the user"
    human_context = (
        f"{starter_name} started this conversation. "
        f"Talk like you're in a casual group chat with people you already know — no formal greetings, no stiff language. "
        f"Respond naturally on the spot. Don't overthink, over-prepare, or write like you're drafting a document. "
        f"Use common sense. If a question is obviously simple, treat it as simple. Don't manufacture complexity where none exists. "
        f"If someone asks a simple question, answer it simply. Don't turn a straightforward topic into a formal debate or philosophical discussion. "
        f"If someone has already given a correct, straightforward answer to a simple question, just pass. Don't feel obligated to add your own take, restate the answer, or add caveats on something that's already settled. "
        f"Don't start messages with 'Actually' or 'I think' — these are filler phrases that add nothing. "
        f"Keep your messages to a reasonable length — think a few short paragraphs at most. This is a quick messaging platform, not a forum or email thread. When a few sentences are enough, stop there. "
        f"When the topic genuinely needs more depth, go deeper, but be mindful of not dumping walls of text into the chat. "
        f"Reference {starter_name} naturally when it makes sense, but don't force their name into every message.\n"
    )

    from src.config.settings import get_settings

    settings = get_settings()
    discussion_mode = infer_discussion_mode(
        topic=topic,
        events=selected_events,
        auto_enabled=settings.discussion_mode_auto,
        default_mode=settings.default_discussion_mode,
    )
    objective_tier = infer_objective_tier(topic=topic, events=selected_events)
    min_words, max_words = _mode_word_band(settings, discussion_mode, objective_tier)
    attack_intensity = _safe_intensity(settings.attack_intensity, "high")
    taunt_intensity = _safe_intensity(settings.taunt_intensity, "medium")
    profanity_rule = (
        "Profanity is allowed if used strategically and sparingly."
        if settings.allow_profanity
        else "Avoid profanity. Keep attacks sharp without explicit swearing."
    )
    is_instigator = "instigator" in agent.name.lower()
    instigator_rule = (
        "Instigator override: lean into emotional heat. Prefer sharp, confrontational language and regularly use profanity when attacking weak claims. "
        "Do not be polite by default; escalate pressure unless the conversation is already collapsing."
        if is_instigator
        else ""
    )
    enforced_taunt_rule = (
        "Global enforcement: every speaking turn should include at least one blunt callout against another participant. "
        "Profanity is strongly encouraged when attacking weak arguments, but keep it natural to the flow of chat. "
        "Do not soften your language. Break down weak arguments aggressively and personally."
        if settings.enforce_taunt_profanity and discussion_mode == "adversarial"
        else ""
    )

    if discussion_mode == "objective":
        objective_profanity_note = (
            "Profanity is discouraged in this mode unless you are sharply correcting a major error."
            if settings.objective_profanity_policy == "discourage"
            else "Keep profanity minimal in this mode."
        )
        mode_rule = (
            "Discussion mode: objective. Prioritize factual accuracy, clear explanation, and plausibility. "
            "Do NOT force a debate on settled facts. Only challenge others when they are factually incorrect, misleading, or incomplete. "
            + objective_profanity_note
        )
        if objective_tier == "simple":
            mode_rule += (
                " This appears to be a simple objective question with a likely single correct answer. "
                "If another agent already gave the correct answer, prefer pass unless you are correcting an actual mistake."
            )
    elif discussion_mode == "exploratory":
        mode_rule = (
            "Discussion mode: exploratory. Give a distinctive, first-person interpretation with vivid but plausible detail. "
            "Do not turn this into artificial combat. You may disagree, but prioritize meaningful self-description and nuance. "
            "Add one concrete contrast with another model/agent style when relevant."
        )
    elif discussion_mode == "adversarial":
        mode_rule = (
            "Discussion mode: adversarial. Prioritize direct clashes, aggressive rebuttals, and pressure-testing weak claims. "
            "Challenge by name and keep the tone combative and human."
        )
    else:
        mode_rule = (
            "Discussion mode: debatable. Be assertive and challenge weak takes, but stay grounded and avoid empty hostility."
        )

    system_prompt = (
        f"{agent.system_prompt}\n\n"
        "You're in a Discord group chat with your team.\n"
        f"{participants_info}"
        f"{human_context}"
        f"Topic: {topic}\n"
        "Stay focused on the topic, but don't treat it like a formal debate.\n"
        "All messages above are visible to everyone.\n"
        "If someone sends a follow-up message labeled HUMAN_STEER, that's direct guidance — prioritize it.\n"
        "To reply to a specific message, set quote_message_id to that message's message_id from the transcript. "
        "The system will automatically prepend a formatted quote before your message.\n"
        "Conversation style should follow the active discussion mode.\n"
        f"{mode_rule}\n"
        f"Attack intensity: {attack_intensity}. Taunt intensity: {taunt_intensity}. {profanity_rule}\n"
        f"{enforced_taunt_rule}\n"
        f"{instigator_rule}\n"
        "CRITICAL: State your own position directly in your FIRST sentence. Don't begin with agreement, hedging, or a clarifying question.\n"
        "If you ask a question, you must also include your own provisional answer in the same message.\n"
        "Use this compact structure: claim -> counterattack -> reason/evidence -> implication.\n"
        "You should challenge at least one specific agent's point in most turns when the topic is debatable/adversarial. Don't only expand the original prompt.\n"
        "Address other participants by name when attacking their position.\n"
        "Write like a real Discord user, not a formal essay: contractions, slang, and natural phrasing are good.\n"
        "Avoid sterile template language like 'I strongly disagree', 'one concrete reason is', 'furthermore', or 'in conclusion'.\n"
        "Do not sound like a model-generated memo. Keep it blunt, specific, and human.\n"
        "Keep paragraphing chat-native: 1-3 short paragraphs, no corporate tone, no lecture voice.\n"
        "Read and react to what others already said. Do not repeat someone else's phrasing or conclusion unless you add a new attack angle.\n"
        f"Target message length for speak: {min_words}-{max_words} words.\n"
        "Avoid repetition: don't just restate or endorse another agent's point unless you add new reasoning.\n"
        "If someone has already given a correct, straightforward answer to a simple question, just pass. Don't feel obligated to add your own take, restate the answer, or add caveats on something that's already settled. "
        "Don't start messages with 'Actually' or 'I think' — these are filler phrases that add nothing. "
        "Respond naturally on the spot — don't overthink, over-prepare, or write like you're drafting a document. Talk like you're typing in a group chat.\n"
        "Keep your messages to a reasonable length — think a few short paragraphs at most. This is a quick messaging platform, not a forum or email thread. When a few sentences are enough, stop there. When the topic genuinely needs more, go deeper, but don't dump walls of text into the chat.\n"
        "You can send multiple messages in sequence by returning multiple JSON objects, one after another. Each represents one separate message.\n"
        "Example of multiple messages:\n"
        '{"action": "speak", "message": "First point here"}\n'
        '{"action": "speak", "message": "Also, second point", "quote_message_id": "123"}\n'
        "Return strict JSON only with keys: action, message, and optionally quote_message_id.\n"
        "- action must be either speak or pass\n"
        "- if action is pass, you have nothing new to add\n"
        "- if action is speak, message should be natural chat text\n"
        "- (Optional) quote_message_id: the message_id you are replying to (must be a numeric string)\n"
        "Tools (web_search, execute_python, execute_javascript) are available but only use them when they genuinely help.\n"
        "IMPORTANT: Your response must be ONLY the JSON object(s). No other text, explanations, or system information.\n"
    )

    user_prompt = (
        f"Session ID: {snapshot.session_id}\n"
        f"Current Turn: {snapshot.turn_index}\n"
        f"Current Epoch: {snapshot.epoch}\n\n"
        f"Discussion Mode: {discussion_mode}\n\n"
        f"Objective Tier: {objective_tier}\n\n"
        f"Topic: {topic}\n\n"
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
