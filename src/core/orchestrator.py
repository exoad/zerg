from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from src.config.settings import Settings
from src.core.agents import AgentConfig, AgentRegistry
from src.core.consensus import detect_consensus
from src.core.context_builder import build_agent_messages, build_moderator_messages
from src.core.discussion_mode import infer_discussion_mode
from src.core.model_assignment import AssignedModel, ModelAssignmentManager
from src.core.session import DebateSession, EventType
from src.llm.router import AgentDecision, LLMRouter


logger = logging.getLogger(__name__)

EmbedPayload = dict[str, Any]
EmitAgentMessage = Callable[[int, AgentConfig, str, list[EmbedPayload] | None], Awaitable[int | None]]
MIN_SESSION_TURNS = 8


@dataclass
class SessionRuntime:
    session: DebateSession
    task: asyncio.Task[None]
    model_assignments: dict[str, AssignedModel] = field(default_factory=dict)
    assignment_epoch: int = -1
    assignment_unique_models: bool = True
    assignment_catalog_errors: dict[str, str] = field(default_factory=dict)
    consecutive_generation_failures: int = 0
    last_failure_notice_count: int = 0


@dataclass(frozen=True)
class SessionRosterEntry:
    agent_name: str
    provider: str
    model: str
    source: str


@dataclass(frozen=True)
class SessionRoster:
    epoch: int
    unique_models: bool
    catalog_errors: dict[str, str]
    entries: list[SessionRosterEntry]


class CouncilOrchestrator:
    def __init__(
        self,
        settings: Settings,
        registry: AgentRegistry,
        router: LLMRouter,
        model_assignment_manager: ModelAssignmentManager,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.router = router
        self.model_assignment_manager = model_assignment_manager

        self._emit: EmitAgentMessage | None = None
        self._sessions: dict[int, SessionRuntime] = {}
        self._lock = asyncio.Lock()

    def set_emitter(self, emit_fn: EmitAgentMessage) -> None:
        self._emit = emit_fn

    async def has_active_session(self, channel_id: int) -> bool:
        async with self._lock:
            runtime = self._sessions.get(channel_id)
            if runtime is None:
                return False
            snapshot = await runtime.session.snapshot()
            return snapshot.active

    async def is_starter(self, channel_id: int, user_id: int) -> bool:
        async with self._lock:
            runtime = self._sessions.get(channel_id)
            if runtime is None:
                return False
            return runtime.session.starter_user_id == user_id

    async def get_session_roster(self, channel_id: int) -> SessionRoster | None:
        async with self._lock:
            runtime = self._sessions.get(channel_id)
            if runtime is None:
                return None

            entries: list[SessionRosterEntry] = []
            for agent in self.registry.agents:
                assigned = runtime.model_assignments.get(
                    agent.name,
                    AssignedModel(
                        provider=agent.provider,
                        model=self._normalize_model_id(agent.provider, agent.model),
                        source="fallback",
                    ),
                )
                entries.append(
                    SessionRosterEntry(
                        agent_name=agent.name,
                        provider=assigned.provider,
                        model=assigned.model,
                        source=assigned.source,
                    )
                )

            epoch = runtime.assignment_epoch if runtime.assignment_epoch >= 0 else runtime.session.epoch
            return SessionRoster(
                epoch=epoch,
                unique_models=runtime.assignment_unique_models,
                catalog_errors=dict(runtime.assignment_catalog_errors),
                entries=entries,
            )

    async def start_session(
        self,
        *,
        channel_id: int,
        starter_user_id: int,
        starter_name: str,
        topic: str,
        max_turns: int | None = None,
    ) -> DebateSession:
        if self._emit is None:
            raise RuntimeError("Emitter callback is not set")

        async with self._lock:
            existing = self._sessions.get(channel_id)
            if existing is not None:
                snapshot = await existing.session.snapshot()
                if snapshot.active:
                    raise ValueError("A council session is already running in this channel")

            requested_turns = max_turns if max_turns is not None else self.settings.council_max_turns
            effective_turns = max(MIN_SESSION_TURNS, requested_turns)
            if effective_turns != requested_turns:
                logger.warning(
                    "Clamped max_turns from %s to minimum %s for channel %s",
                    requested_turns,
                    MIN_SESSION_TURNS,
                    channel_id,
                )

            session = DebateSession(
                session_id=str(uuid.uuid4()),
                channel_id=channel_id,
                starter_user_id=starter_user_id,
                starter_display_name=starter_name,
                topic=topic.strip(),
                max_turns=effective_turns,
            )
            await session.start(starter_name, topic)
            logger.info(
                "Started session %s in channel %s with max_turns=%s (requested=%s)",
                session.session_id,
                channel_id,
                session.max_turns,
                requested_turns,
            )

            initial_assignments: dict[str, AssignedModel]
            catalog_errors: dict[str, str]
            unique_models: bool
            try:
                assignment_set = await self.model_assignment_manager.assign(self.registry.agents)
                initial_assignments = dict(assignment_set.by_agent)
                catalog_errors = dict(assignment_set.catalog_errors)
                unique_models = assignment_set.used_unique_models
                if catalog_errors:
                    logger.warning(
                        "Model catalog returned provider errors during startup reshuffle: %s",
                        catalog_errors,
                    )
                if not unique_models:
                    logger.warning("Startup reshuffle could not keep fully unique models")
            except Exception as exc:
                logger.warning("Startup model reshuffle failed, using configured defaults: %s", exc)
                initial_assignments = {
                    agent.name: AssignedModel(
                        provider=agent.provider,
                        model=self._normalize_model_id(agent.provider, agent.model),
                        source="fallback",
                    )
                    for agent in self.registry.agents
                }
                catalog_errors = {"system": str(exc)}
                unique_models = False

            task = asyncio.create_task(self._run_session_loop(channel_id))
            self._sessions[channel_id] = SessionRuntime(
                session=session,
                task=task,
                model_assignments=initial_assignments,
                assignment_epoch=session.epoch,
                assignment_unique_models=unique_models,
                assignment_catalog_errors=catalog_errors,
            )
            return session

    async def add_human_steer(
        self,
        *,
        channel_id: int,
        user_id: int,
        user_name: str,
        text: str,
    ) -> bool:
        runtime = await self._get_runtime(channel_id)
        if runtime is None:
            return False
        if runtime.session.starter_user_id != user_id:
            return False

        await runtime.session.add_human_input(user_id=user_id, display_name=user_name, content=text)
        return True

    async def stop_session(self, channel_id: int, reason: str) -> None:
        runtime = await self._get_runtime(channel_id)
        if runtime is None:
            return

        await runtime.session.close(reason)
        runtime.task.cancel()

    async def _get_runtime(self, channel_id: int) -> SessionRuntime | None:
        async with self._lock:
            return self._sessions.get(channel_id)

    async def _run_session_loop(self, channel_id: int) -> None:
        runtime = await self._get_runtime(channel_id)
        if runtime is None:
            return

        session = runtime.session
        started_at = datetime.now(timezone.utc)
        agent_next_allowed_at: dict[str, float] = {agent.name: 0.0 for agent in self.registry.agents}
        outbound_history: deque[float] = deque()
        global_cap = max(1, self.settings.global_max_messages_per_minute)
        bootstrap_epoch = -1
        required_bootstrap = max(1, min(len(self.registry.agents), self.settings.steer_bootstrap_required_agents))
        bootstrap_targets = [agent.name for agent in self.registry.agents[:required_bootstrap]]
        bootstrap_pending: set[str] = set()
        pending_rebuttal_targets: set[str] = set()
        last_speaker: str | None = None
        root_to_agent = self._root_to_agent_map(self.registry.agents)
        instigator_agent = next((a for a in self.registry.agents if "instigator" in a.name.lower()), None)
        messages_since_instigator = 0
        instigator_interval = 3

        try:
            while True:
                snapshot = await session.snapshot()
                if not snapshot.active:
                    logger.info("Session became inactive; exiting loop for channel %s", channel_id)
                    break

                if snapshot.epoch != bootstrap_epoch:
                    bootstrap_epoch = snapshot.epoch
                    bootstrap_pending = set(bootstrap_targets)
                    pending_rebuttal_targets.clear()
                    last_speaker = None
                    logger.info(
                        "Starting stance bootstrap for epoch %s in channel %s (required=%s)",
                        bootstrap_epoch,
                        channel_id,
                        len(bootstrap_pending),
                    )

                discussion_mode = infer_discussion_mode(
                    topic=snapshot.topic,
                    events=snapshot.events,
                    auto_enabled=self.settings.discussion_mode_auto,
                    default_mode=self.settings.default_discussion_mode,
                )

                force_finalize = await session.should_force_finalize()
                timed_out = self._timed_out(started_at)
                if force_finalize or timed_out:
                    reason = "max_turns" if force_finalize else "timeout"
                    logger.info(
                        "Finalizing session for channel %s due to %s (max_turns=%s turn_index=%s)",
                        channel_id,
                        reason,
                        session.max_turns,
                        snapshot.turn_index,
                    )
                    await self._finalize(runtime, force=True, started_at=started_at)
                    break

                await self._refresh_model_assignments_if_needed(runtime, snapshot.epoch)
                await session.pop_interrupt()

                now = time.monotonic()
                all_eligible_agents = [
                    agent for agent in self.registry.agents if now >= agent_next_allowed_at.get(agent.name, 0.0)
                ]

                if bootstrap_pending:
                    eligible_agents = [agent for agent in all_eligible_agents if agent.name in bootstrap_pending]
                else:
                    eligible_agents = all_eligible_agents

                if (
                    not bootstrap_pending
                    and instigator_agent is not None
                    and discussion_mode == "adversarial"
                    and messages_since_instigator >= instigator_interval
                    and any(a.name == instigator_agent.name for a in eligible_agents)
                ):
                    eligible_agents = [instigator_agent]

                if (
                    not bootstrap_pending
                    and self.settings.rebuttal_targeting_enabled
                    and discussion_mode in {"debatable", "adversarial"}
                    and eligible_agents
                ):
                    targeted = [agent for agent in eligible_agents if agent.name in pending_rebuttal_targets]
                    if targeted:
                        eligible_agents = targeted
                    elif last_speaker and len(eligible_agents) > 1:
                        no_repeat = [agent for agent in eligible_agents if agent.name != last_speaker]
                        if no_repeat:
                            eligible_agents = no_repeat

                if not eligible_agents:
                    if bootstrap_pending:
                        soonest = min(
                            (agent_next_allowed_at.get(name, now + 0.25) for name in bootstrap_pending),
                            default=now + 0.25,
                        )
                    else:
                        soonest = min(agent_next_allowed_at.values()) if agent_next_allowed_at else now + 0.25
                    wait_for = max(0.05, min(1.0, soonest - now))
                    await asyncio.sleep(wait_for)
                    continue

                epoch_at_start = snapshot.epoch
                tasks_by_handle: dict[asyncio.Task[AgentDecision], AgentConfig] = {}
                assigned_by_name: dict[str, AssignedModel] = {}
                for agent in eligible_agents:
                    assigned = runtime.model_assignments.get(
                        agent.name,
                        AssignedModel(
                            provider=agent.provider,
                            model=self._normalize_model_id(agent.provider, agent.model),
                            source="fallback",
                        ),
                    )
                    assigned_by_name[agent.name] = assigned
                    other_agents = [a for a in self.registry.agents if a.name != agent.name]
                    messages = build_agent_messages(
                        snapshot,
                        agent,
                        other_agents=other_agents,
                        max_events=self.settings.max_context_events,
                        max_chars=self.settings.max_context_chars,
                    )
                    handle = asyncio.create_task(
                        self.router.decide(
                            agent,
                            messages,
                            timeout_sec=self.settings.agent_response_timeout_sec,
                            provider_override=assigned.provider,
                            model_override=assigned.model,
                        )
                    )
                    tasks_by_handle[handle] = agent

                interrupt_task = asyncio.create_task(session.interrupt_event.wait())
                done, pending = await asyncio.wait(
                    {interrupt_task, *tasks_by_handle.keys()},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if interrupt_task in done and session.interrupt_event.is_set():
                    for task in tasks_by_handle.keys():
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks_by_handle.keys(), return_exceptions=True)
                    await session.pop_interrupt()
                    continue

                interrupt_task.cancel()
                await asyncio.gather(interrupt_task, return_exceptions=True)

                winner_task = next((task for task in done if task is not interrupt_task), None)
                winner_agent = tasks_by_handle.get(winner_task) if winner_task else None

                extra_done = [task for task in done if task is not interrupt_task and task is not winner_task]
                for task in extra_done:
                    try:
                        _ = task.result()
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass

                for task in pending:
                    if task is not interrupt_task:
                        task.cancel()
                pending_non_interrupt = [task for task in pending if task is not interrupt_task]
                if pending_non_interrupt:
                    await asyncio.gather(*pending_non_interrupt, return_exceptions=True)

                if winner_task is None or winner_agent is None:
                    await asyncio.sleep(0.05)
                    continue

                try:
                    decision = winner_task.result()
                except asyncio.CancelledError:
                    continue
                except Exception as exc:
                    runtime.consecutive_generation_failures += 1
                    logger.warning("Agent generation failed for %s: %s", winner_agent.name, exc)
                    await self._maybe_emit_generation_failure_notice(runtime, channel_id, winner_agent.name, exc)

                    fail_meta = {
                        "provider": assigned_by_name[winner_agent.name].provider,
                        "model": assigned_by_name[winner_agent.name].model,
                        "assignment_source": assigned_by_name[winner_agent.name].source,
                        "assignment_epoch": runtime.assignment_epoch,
                        "assignment_unique_models": runtime.assignment_unique_models,
                    }
                    await session.add_agent_pass(winner_agent.name, "pass due to transient model failure", metadata=fail_meta)
                    current_epoch = (await session.snapshot()).epoch
                    if current_epoch == epoch_at_start and winner_agent.name in bootstrap_pending:
                        bootstrap_pending.discard(winner_agent.name)
                    agent_next_allowed_at[winner_agent.name] = time.monotonic() + self.settings.agent_min_cooldown_seconds
                    continue

                runtime.consecutive_generation_failures = 0
                assigned = assigned_by_name[winner_agent.name]
                logger.info(
                    'Agent %s decision: action="%s" quote="%s" provider="%s" model="%s" extra_messages=%d raw="%s"',
                    winner_agent.name,
                    decision.action,
                    decision.quote_message_id or "",
                    decision.provider,
                    decision.model,
                    len(decision.multi_messages),
                    decision.raw_text.replace('"', '\\"'),
                )
                if decision.tool_results:
                    logger.info(
                        'Agent %s tool_results:\n"%s"',
                        winner_agent.name,
                        decision.tool_results.replace('"', '\\"'),
                    )

                post_generation = await session.snapshot()
                stale = post_generation.epoch != epoch_at_start

                tool_embed: list[EmbedPayload] | None = None
                if decision.tool_results:
                    tool_embed = [{
                        "title": "Tool Execution",
                        "description": decision.tool_results[:1000],
                        "color": 0x5865F2,
                    }]

                messages_to_send: list[AgentDecision] = [decision, *decision.multi_messages]
                MAX_MESSAGES_PER_DECISION = 3
                if len(messages_to_send) > MAX_MESSAGES_PER_DECISION:
                    logger.info(
                        "Agent %s wanted to send %d messages, capping to %d",
                        winner_agent.name,
                        len(messages_to_send),
                        MAX_MESSAGES_PER_DECISION,
                    )
                    messages_to_send = messages_to_send[:MAX_MESSAGES_PER_DECISION]

                sent_visible_message = False
                for i, sub_decision in enumerate(messages_to_send):
                    if i > 0:
                        await asyncio.sleep(0.6)

                    sub_message = sub_decision.message
                    quote_target_id = sub_decision.quote_message_id
                    if quote_target_id:
                        target_event = next(
                            (
                                e
                                for e in reversed(post_generation.events)
                                if str(e.metadata.get("discord_message_id")) == quote_target_id
                            ),
                            None,
                        )
                        if target_event:
                            speaker = target_event.actor_name or "Unknown"
                            quoted = target_event.content.replace("\n", " ").strip()
                            if len(quoted) > 120:
                                quoted = quoted[:117].rstrip() + "..."
                            sub_message = f"> {speaker}: {quoted}\n\n{sub_message}"

                    sub_meta: dict[str, Any] = {
                        "latency_ms": round(sub_decision.latency_ms, 2),
                        "provider": sub_decision.provider,
                        "model": sub_decision.model,
                        "attempts_used": sub_decision.attempts_used,
                        "tools_used": dict(sub_decision.tools_used),
                        "assignment_source": assigned.source,
                        "assignment_epoch": runtime.assignment_epoch,
                        "assignment_unique_models": runtime.assignment_unique_models,
                    }
                    if quote_target_id:
                        sub_meta["quote_message_id"] = quote_target_id

                    if stale:
                        await session.add_agent_message(winner_agent.name, sub_message, stale=True, metadata=sub_meta)
                        break

                    if sub_decision.action == "pass":
                        await session.add_agent_pass(winner_agent.name, sub_decision.message, metadata=sub_meta)
                    else:
                        await self._wait_for_global_message_slot(outbound_history, global_cap)
                        msg_id = await self._emit_agent(
                            channel_id,
                            winner_agent,
                            sub_message,
                            embed_payloads=tool_embed if i == 0 else None,
                        )
                        if msg_id:
                            sub_meta["discord_message_id"] = msg_id
                            outbound_history.append(time.monotonic())
                            sent_visible_message = True
                        else:
                            sub_meta["delivery_failed"] = True
                            logger.warning(
                                "Agent %s generated a message but delivery failed (channel=%s, action=%s, len=%d)",
                                winner_agent.name,
                                channel_id,
                                sub_decision.action,
                                len(sub_message),
                            )
                        await session.add_agent_message(winner_agent.name, sub_message, stale=False, metadata=sub_meta)

                    current = await session.snapshot()
                    stale = current.epoch != epoch_at_start
                    if stale:
                        break

                agent_next_allowed_at[winner_agent.name] = time.monotonic() + self.settings.agent_min_cooldown_seconds
                pending_rebuttal_targets.discard(winner_agent.name)
                if not stale and winner_agent.name in bootstrap_pending:
                    bootstrap_pending.discard(winner_agent.name)
                    if not bootstrap_pending:
                        logger.info("Completed stance bootstrap for epoch %s in channel %s", bootstrap_epoch, channel_id)

                if sent_visible_message:
                    last_speaker = winner_agent.name
                    if instigator_agent is not None and winner_agent.name == instigator_agent.name:
                        messages_since_instigator = 0
                    else:
                        messages_since_instigator += 1

                if self.settings.rebuttal_targeting_enabled and discussion_mode in {"debatable", "adversarial"} and decision.action == "speak":
                    targets = self._extract_targeted_agents(decision.message, root_to_agent)
                    for target in targets:
                        if target != winner_agent.name:
                            pending_rebuttal_targets.add(target)

                latest = await session.snapshot()
                if not bootstrap_pending:
                    if detect_consensus(
                        list(latest.events),
                        self.settings.consensus_threshold,
                        min_recent_messages=self.settings.consensus_min_recent_messages,
                        min_unique_agents=min(self.settings.consensus_min_unique_agents, len(self.registry.agents)),
                        min_turns=self.settings.consensus_min_turns,
                        pass_count_required=self.settings.consensus_pass_count_required,
                        lexical_weight=self.settings.consensus_lexical_weight,
                        stance_weight=self.settings.consensus_stance_weight,
                        freshness_weight=self.settings.consensus_freshness_weight,
                        disagreement_guard_enabled=self.settings.consensus_disagreement_guard_enabled,
                    ):
                        await self._finalize(runtime, force=False, started_at=started_at)
                        break

                if sent_visible_message:
                    await asyncio.sleep(max(0.0, self.settings.message_delay_seconds))
        except asyncio.CancelledError:
            logger.info("Session loop canceled for channel %s", channel_id)
            raise
        finally:
            async with self._lock:
                current = self._sessions.get(channel_id)
                if current is runtime:
                    del self._sessions[channel_id]

    def _timed_out(self, started_at: datetime) -> bool:
        cutoff = timedelta(minutes=self.settings.force_finalize_timeout_minutes)
        return datetime.now(timezone.utc) - started_at >= cutoff

    async def _finalize(self, runtime: SessionRuntime, *, force: bool, started_at: datetime) -> None:
        snapshot = await runtime.session.snapshot()

        moderator = self.registry.moderator
        m = self.settings.consensus_model or moderator.model
        
        # Normalize the model string so empty ones fallback successfully onto the curated list
        model = self._normalize_model_id(moderator.provider, m)
        provider = self._provider_from_model(model, fallback=moderator.provider)
        
        messages = build_moderator_messages(
            snapshot,
            moderator,
            force=force,
            max_chars=self.settings.max_context_chars,
        )

        try:
            summary = await self.router.summarize(
                provider=provider,
                model=model,
                messages=messages,
                timeout_sec=self.settings.agent_response_timeout_sec,
            )
        except Exception as exc:
            if self.router.is_fatal_model_error(exc):
                logger.warning("Consensus model summary failed (%s), falling back to default moderator model.", type(exc).__name__)
                try:
                    fallback_model = self._normalize_model_id(moderator.provider, moderator.model)
                    fallback_provider = self._provider_from_model(fallback_model, fallback=moderator.provider)
                    summary = await self.router.summarize(
                        provider=fallback_provider,
                        model=fallback_model,
                        messages=messages,
                        timeout_sec=self.settings.agent_response_timeout_sec,
                    )
                except Exception as inner_exc:
                    logger.warning("Moderator fallback summary failed: %s", inner_exc)
                    summary = (
                        "The council has ended due to limits. Please review the latest agent messages "
                        "and restart with a narrower steer if needed. DEBATE CLOSED."
                    )
            else:
                logger.warning("Moderator summary failed: %s", exc)
                summary = (
                    "The council has ended due to limits. Please review the latest agent messages "
                    "and restart with a narrower steer if needed. DEBATE CLOSED."
                )

        if force:
            await runtime.session.mark_forced_finalize(summary)
        else:
            await runtime.session.mark_consensus(summary)

        final_snapshot = await runtime.session.snapshot()
        embeds = self._build_consensus_embed_payloads(
            final_snapshot,
            summary=summary,
            force=force,
            started_at=started_at,
        )
        await self._emit_agent(snapshot.channel_id, moderator, "", embed_payloads=embeds)

    async def _emit_agent(
        self,
        channel_id: int,
        agent: AgentConfig,
        content: str,
        *,
        embed_payloads: list[EmbedPayload] | None = None,
    ) -> int | None:
        if self._emit is None:
            return None

        has_content = bool(content.strip())
        if not has_content and not embed_payloads:
            return None

        clipped = content.strip() if has_content else ""
        if clipped and len(clipped) > 2000:
            clipped = clipped[:1997] + "..."

        return await self._emit(channel_id, agent, clipped, embed_payloads)

    async def _refresh_model_assignments_if_needed(self, runtime: SessionRuntime, epoch: int) -> None:
        if runtime.assignment_epoch == epoch and runtime.model_assignments:
            return

        try:
            assignment_set = await self.model_assignment_manager.assign(self.registry.agents)
            runtime.model_assignments = dict(assignment_set.by_agent)
            runtime.assignment_catalog_errors = dict(assignment_set.catalog_errors)
            runtime.assignment_unique_models = assignment_set.used_unique_models
            runtime.assignment_epoch = epoch
            if assignment_set.catalog_errors:
                logger.warning(
                    "Model catalog returned provider errors during reshuffle: %s",
                    assignment_set.catalog_errors,
                )
            if not assignment_set.used_unique_models:
                logger.warning("Reshuffle could not keep fully unique models; fallback duplication applied")
        except Exception as exc:
            logger.warning("Model reshuffle failed, using configured defaults: %s", exc)
            runtime.model_assignments = {
                agent.name: AssignedModel(
                    provider=agent.provider,
                    model=self._normalize_model_id(agent.provider, agent.model),
                    source="fallback",
                )
                for agent in self.registry.agents
            }
            runtime.assignment_catalog_errors = {"system": str(exc)}
            runtime.assignment_unique_models = False
            runtime.assignment_epoch = epoch

    async def _maybe_emit_generation_failure_notice(
        self,
        runtime: SessionRuntime,
        channel_id: int,
        failed_agent_name: str,
        exc: Exception,
    ) -> None:
        threshold = max(3, len(self.registry.agents) // 2)
        failures = runtime.consecutive_generation_failures
        if failures < threshold:
            return

        if failures != threshold and (failures - runtime.last_failure_notice_count) < threshold:
            return

        runtime.last_failure_notice_count = failures
        details = str(exc).replace("\n", " ").strip()
        if details.startswith("{") or details.startswith("["):
            details = f"{type(exc).__name__} (see logs for provider payload details)"
        details = details.replace("```", "").strip()
        if len(details) > 220:
            details = details[:217] + "..."

        catalog_hint = ""
        if runtime.assignment_catalog_errors:
            providers = ", ".join(sorted(runtime.assignment_catalog_errors.keys()))
            catalog_hint = f" Catalog issues: {providers}."

        model_hint = ""
        assigned = runtime.model_assignments.get(failed_agent_name)
        if assigned is not None:
            model_hint = f" Assigned model: {assigned.model} (source={assigned.source})."

        message = (
            "Council status: generation is failing repeatedly "
            f"({failures} consecutive failures). "
            f"Last failure: {failed_agent_name}."
            f"{model_hint}"
            f"{catalog_hint}"
            f" Error: {details}"
        )
        await self._emit_agent(channel_id, self.registry.moderator, message, embed_payloads=None)

    @staticmethod
    async def _wait_for_global_message_slot(outbound_history: deque[float], global_cap: int) -> None:
        while True:
            now = time.monotonic()
            while outbound_history and now - outbound_history[0] >= 60.0:
                outbound_history.popleft()
            if len(outbound_history) < global_cap:
                return
            wait_for = 60.0 - (now - outbound_history[0])
            await asyncio.sleep(max(0.05, min(1.0, wait_for)))

    @staticmethod
    def _root_to_agent_map(agents: list[AgentConfig]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for agent in agents:
            root = agent.name.split("(")[0].strip().lower()
            if root:
                mapping[root] = agent.name
        return mapping

    @staticmethod
    def _extract_targeted_agents(message: str, root_to_agent: dict[str, str]) -> set[str]:
        lowered = message.lower()
        targets: set[str] = set()
        for root, full_name in root_to_agent.items():
            if root and f"{root}" in lowered:
                targets.add(full_name)
        return targets

    @staticmethod
    def _normalize_model_id(provider: str, model: str) -> str:
        cleaned_provider = provider.lower().strip() if provider else "groq"
        cleaned_model = model.strip() if model else ""
        
        # Provide a default curated model if the string is empty
        if not cleaned_model:
            from src.core.model_assignment import CURATED_MODEL_POOLS
            pool = CURATED_MODEL_POOLS.get(cleaned_provider)
            cleaned_model = pool[0] if pool else "llama-3.1-8b-instant"
            
        if "/" in cleaned_model:
            return cleaned_model
        return f"{cleaned_provider}/{cleaned_model}"

    def _build_consensus_embed_payloads(
        self,
        snapshot,
        *,
        summary: str,
        force: bool,
        started_at: datetime,
    ) -> list[EmbedPayload]:
        duration = datetime.now(timezone.utc) - started_at
        duration_sec = int(duration.total_seconds())

        participants: set[str] = set()
        providers: dict[str, int] = {}
        per_agent: dict[str, dict[str, Any]] = {}

        for event in snapshot.events:
            if event.event_type not in {EventType.AGENT_MESSAGE, EventType.AGENT_STALE, EventType.AGENT_PASS}:
                continue

            if not event.actor_name:
                continue

            name = event.actor_name
            participants.add(name)
            info = per_agent.setdefault(
                name,
                {
                    "messages": 0,
                    "passes": 0,
                    "latencies": [],
                    "provider": "unknown",
                    "model": "unknown",
                    "attempts": [],
                    "tools_used": {},
                },
            )

            provider = str(event.metadata.get("provider", info["provider"]))
            model = str(event.metadata.get("model", info["model"]))
            info["provider"] = provider
            info["model"] = model
            
            tools_used = event.metadata.get("tools_used", {})
            for tn, tc in tools_used.items():
                info["tools_used"][tn] = info["tools_used"].get(tn, 0) + tc

            if provider and provider != "unknown":
                providers[provider] = providers.get(provider, 0) + 1

            if event.event_type in {EventType.AGENT_MESSAGE, EventType.AGENT_STALE}:
                info["messages"] += 1
            if event.event_type == EventType.AGENT_PASS:
                info["passes"] += 1

            latency_ms = float(event.metadata.get("latency_ms", 0.0) or 0.0)
            if latency_ms > 0:
                info["latencies"].append(latency_ms)

            attempts_used = int(event.metadata.get("attempts_used", 0) or 0)
            if attempts_used > 0:
                info["attempts"].append(attempts_used)

        sorted_participants = sorted(participants)
        providers_text = ", ".join(f"{provider}:{count}" for provider, count in sorted(providers.items())) or "n/a"

        summary_embed: EmbedPayload = {
            "title": "Council Final Summary",
            "description": summary[:4000],
            "color": 0xE67E22 if force else 0x2ECC71,
            "fields": [
                {"name": "Finalize Mode", "value": "Forced" if force else "Consensus", "inline": True},
                {"name": "Total Duration", "value": f"{duration_sec}s", "inline": True},
                {"name": "Total Turns", "value": str(snapshot.turn_index), "inline": True},
                {
                    "name": "Participated Agents",
                    "value": ", ".join(sorted_participants)[:1000] or "n/a",
                    "inline": False,
                },
                {"name": "Provider Activity", "value": providers_text[:1000], "inline": False},
            ],
            "footer": "DEBATE CLOSED",
        }

        performance_fields: list[dict[str, Any]] = []
        for agent_name in sorted(per_agent.keys()):
            data = per_agent[agent_name]
            latencies = data["latencies"]
            avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
            max_latency = max(latencies) if latencies else 0.0
            attempts = data["attempts"]
            avg_attempts = (sum(attempts) / len(attempts)) if attempts else 0.0

            value = (
                f"provider: {data['provider']}\n"
                f"model: {data['model']}\n"
                f"messages: {data['messages']} | passes: {data['passes']}\n"
                f"avg latency: {avg_latency:.0f} ms | max latency: {max_latency:.0f} ms\n"
                f"avg attempts: {avg_attempts:.2f}"
            )
            tools_used = data.get("tools_used", {})
            if tools_used:
                tools_str = ", ".join(f"{t}:{c}" for t, c in tools_used.items())
                value += f"\n🛠️ Tools: {tools_str}"
            performance_fields.append({"name": agent_name[:256], "value": value[:1024], "inline": False})

        performance_embed: EmbedPayload = {
            "title": "Consensus Debug Details",
            "color": 0x3498DB,
            "fields": performance_fields[:25],
        }

        return [summary_embed, performance_embed]

    @staticmethod
    def _provider_from_model(model: str, fallback: str) -> str:
        if "/" in model:
            return model.split("/", 1)[0].lower()
        return fallback.lower()
