from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from litellm import acompletion

from src.config.settings import Settings
from src.core.agents import AgentConfig
from src.llm.rate_limits import RateLimitManager
from src.tools.registry import get_tool_schema, dispatch_tool


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentDecision:
    action: str
    message: str
    raw_text: str
    latency_ms: float
    provider: str
    model: str
    attempts_used: int
    quote_message_id: str | None = None
    tools_used: dict[str, int] = field(default_factory=dict)
    tool_results: str | None = None
    multi_messages: list["AgentDecision"] = field(default_factory=list)


@dataclass(frozen=True)
class CompletionResult:
    text: str
    latency_ms: float
    attempts_used: int
    tools_used: dict[str, int] = field(default_factory=dict)
    tool_results_text: str | None = None


class LLMRouter:
    MAX_RETRY_AFTER_SECONDS = 300.0

    def __init__(self, settings: Settings, rate_limits: RateLimitManager) -> None:
        self.settings = settings
        self.rate_limits = rate_limits

    @staticmethod
    def _normalize_model_id(provider: str, model: str | None) -> str:
        cleaned_provider = provider.lower().strip() if provider else "groq"
        cleaned_model = model.strip() if model else ""
        if not cleaned_model:
            from src.core.model_assignment import CURATED_MODEL_POOLS

            pool = CURATED_MODEL_POOLS.get(cleaned_provider)
            cleaned_model = pool[0] if pool else "llama-3.1-8b-instant"
        if "/" in cleaned_model:
            return cleaned_model
        return f"{cleaned_provider}/{cleaned_model}"

    async def decide(
        self,
        agent: AgentConfig,
        messages: list[dict[str, Any]],
        *,
        timeout_sec: int,
        provider_override: str | None = None,
        model_override: str | None = None,
    ) -> AgentDecision:
        provider = (provider_override or agent.provider).lower().strip()
        model = self._normalize_model_id(provider, model_override or agent.model)
        actual_provider = provider
        actual_model = model
        try:
            result = await self._complete_with_retries(
                provider=provider,
                model=model,
                messages=messages,
                timeout_sec=timeout_sec,
                max_tokens=self.settings.agent_response_max_tokens,
            )
        except Exception as exc:
            if model_override and self.is_fatal_model_error(exc):
                fallback_provider = agent.provider.lower().strip()
                fallback_model = self._normalize_model_id(fallback_provider, agent.model)
                if (fallback_provider, fallback_model) != (provider, model):
                    logger.warning(
                        "Model override %s/%s unusable for %s (%s); retrying with agent default %s/%s",
                        provider,
                        model,
                        agent.name,
                        type(exc).__name__,
                        fallback_provider,
                        fallback_model,
                    )
                    actual_provider = fallback_provider
                    actual_model = fallback_model
                    result = await self._complete_with_retries(
                        provider=fallback_provider,
                        model=fallback_model,
                        messages=messages,
                        timeout_sec=timeout_sec,
                        max_tokens=self.settings.agent_response_max_tokens,
                    )
                else:
                    logger.warning(
                        "Fatal model error for %s with provider=%s model=%s: %s",
                        agent.name,
                        provider,
                        model,
                        type(exc).__name__,
                    )
                    raise
            else:
                if self.is_fatal_model_error(exc):
                    logger.warning(
                        "Fatal model error for %s with provider=%s model=%s: %s",
                        agent.name,
                        provider,
                        model,
                        type(exc).__name__,
                    )
                raise
        logger.info('Agent %s using model "%s" provider "%s" produced output: "%s"', agent.name, actual_model, actual_provider, result.text.replace('"','\'"\''))
        recent_contents, recent_speakers = self._extract_recent_agent_context(messages)
        discussion_mode = self._extract_discussion_mode(messages)
        objective_tier = self._extract_objective_tier(messages)
        is_instigator = "instigator" in agent.name.lower()
        dec = self._parse_decision(
            result.text,
            latency_ms=result.latency_ms,
            provider=actual_provider,
            model=actual_model,
            attempts_used=result.attempts_used,
        )

        if discussion_mode == "objective" and objective_tier == "simple" and dec.action == "speak":
            correction_count = self._objective_correction_count(recent_contents)
            is_correction = self._is_correction_message(dec.message)
            if is_correction and correction_count >= self.settings.objective_simple_max_corrections:
                dec = AgentDecision(
                    action="pass",
                    message="pass",
                    raw_text=dec.raw_text,
                    latency_ms=dec.latency_ms,
                    provider=dec.provider,
                    model=dec.model,
                    attempts_used=dec.attempts_used,
                    quote_message_id=dec.quote_message_id,
                )
            elif (not is_correction) and self._is_redundant_objective_answer(dec.message, recent_contents):
                dec = AgentDecision(
                    action="pass",
                    message="pass",
                    raw_text=dec.raw_text,
                    latency_ms=dec.latency_ms,
                    provider=dec.provider,
                    model=dec.model,
                    attempts_used=dec.attempts_used,
                    quote_message_id=dec.quote_message_id,
                )

        needs_rewrite, rewrite_reason = self._needs_rewrite(
            dec,
            recent_contents,
            recent_speakers,
            discussion_mode=discussion_mode,
            objective_tier=objective_tier,
            is_instigator=is_instigator,
        )
        if (
            self.settings.stance_retry_enabled
            and dec.action == "speak"
            and needs_rewrite
        ):
            logger.info("Rewriting first draft for %s due to: %s", agent.name, rewrite_reason)
            rewrite_messages = [
                *messages,
                {"role": "assistant", "content": result.text},
                {
                    "role": "user",
                    "content": (
                        "Rewrite your last answer with a direct stance-first response. "
                        f"Issue to fix: {rewrite_reason}. "
                        f"Current discussion mode: {discussion_mode}. "
                        f"Current objective tier: {objective_tier}. "
                        "Write like an actual Discord user, not a formal assistant. "
                        "Rules: (1) first sentence must state your position, "
                        + (
                            "(2) include at least one concrete reason and one pointed challenge to another agent's claim, "
                            if discussion_mode != "objective"
                            else "(2) include at least one concrete factual reason; only challenge another agent if they are factually wrong, "
                        )
                        + "(3) if you ask a question, include your own provisional answer, "
                        + f"(4) target {self._word_bounds_for_mode(discussion_mode, objective_tier)[0]}-{self._word_bounds_for_mode(discussion_mode, objective_tier)[1]} words, "
                        + "(5) do not paraphrase previous messages. Add new attack angle/evidence. "
                        + (
                            "(6) include at least one direct taunt by name and at least one profanity term. "
                            if self.settings.enforce_taunt_profanity and discussion_mode == "adversarial"
                            else ""
                        )
                        + (
                            "(7) for your instigator role, prefer emotionally aggressive language and frequent profanity when it sharpens your attack. "
                            if is_instigator
                            else ""
                        )
                        + "(8) avoid robotic templates like 'I strongly disagree', 'one concrete reason is', 'furthermore', 'in conclusion'. "
                        "Return strict JSON only with action/message and optional quote_message_id."
                    ),
                },
            ]
            try:
                rewrite = await self._complete_with_retries(
                    provider=actual_provider,
                    model=actual_model,
                    messages=rewrite_messages,
                    timeout_sec=timeout_sec,
                    max_tokens=self.settings.agent_response_max_tokens,
                )
                logger.info(
                    'Agent %s rewrite output: "%s"',
                    agent.name,
                    rewrite.text.replace('"', '\'"\''),
                )
                rewritten = self._parse_decision(
                    rewrite.text,
                    latency_ms=(result.latency_ms + rewrite.latency_ms),
                    provider=actual_provider,
                    model=actual_model,
                    attempts_used=(result.attempts_used + rewrite.attempts_used),
                )
                rewrite_still_bad, _ = self._needs_rewrite(
                    rewritten,
                    recent_contents,
                    recent_speakers,
                    discussion_mode=discussion_mode,
                    objective_tier=objective_tier,
                    is_instigator=is_instigator,
                )
                if rewritten.action == "speak" and rewrite_still_bad:
                    if is_instigator:
                        dec = rewritten
                    else:
                        dec = AgentDecision(
                            action="pass",
                            message="pass",
                            raw_text=rewritten.raw_text,
                            latency_ms=rewritten.latency_ms,
                            provider=rewritten.provider,
                            model=rewritten.model,
                            attempts_used=rewritten.attempts_used,
                        )
                else:
                    dec = rewritten

                merged_tools = dict(result.tools_used)
                for tool_name, count in rewrite.tools_used.items():
                    merged_tools[tool_name] = merged_tools.get(tool_name, 0) + count

                result = CompletionResult(
                    text=dec.raw_text,
                    latency_ms=dec.latency_ms,
                    attempts_used=dec.attempts_used,
                    tools_used=merged_tools,
                    tool_results_text="\n".join(
                        [txt for txt in [result.tool_results_text, rewrite.tool_results_text] if txt]
                    )
                    or None,
                )
            except Exception as exc:
                logger.warning("Stance rewrite failed for %s: %s", agent.name, exc)

        return AgentDecision(
            action=dec.action,
            message=dec.message,
            raw_text=dec.raw_text,
            latency_ms=dec.latency_ms,
            provider=dec.provider,
            model=dec.model,
            attempts_used=dec.attempts_used,
            quote_message_id=dec.quote_message_id,
            tools_used=result.tools_used,
            tool_results=result.tool_results_text,
        )

    @staticmethod
    def _word_count(message: str) -> int:
        return len(re.findall(r"\b\w+\b", message))

    def _is_low_substance_message(self, message: str) -> bool:
        text = " ".join(message.strip().split())
        if not text:
            return True

        lowered = text.lower()
        word_count = self._word_count(text)
        if word_count < 12:
            return True

        if re.match(
            r"^(i agree|agree|i see .* point|good point|that's fair|you'?re right|i think .* point|that'?s a good point)\b",
            lowered,
        ):
            if not re.search(r"\b(because|my view|i believe|therefore|so)\b", lowered):
                return True

        question_marks = text.count("?")
        sentence_marks = len(re.findall(r"[.!?]", text))
        if question_marks > 0:
            has_claim_marker = re.search(
                r"\b(i believe|my view|the answer is|it is|it means|because|therefore|in practice|i conclude)\b",
                lowered,
            )
            mostly_questions = sentence_marks <= question_marks + 1
            if mostly_questions and not has_claim_marker:
                return True

        return False

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9']+", text.lower()) if len(token) > 2}

    def _max_similarity_to_recent(self, message: str, recent_contents: Iterable[str]) -> float:
        base = self._token_set(message)
        if not base:
            return 0.0
        best = 0.0
        for other in recent_contents:
            tokens = self._token_set(other)
            if not tokens:
                continue
            union = len(base | tokens)
            if union == 0:
                continue
            score = len(base & tokens) / union
            if score > best:
                best = score
        return best

    @staticmethod
    def _display_name_roots(agent_names: Iterable[str]) -> list[str]:
        roots: list[str] = []
        for name in agent_names:
            head = name.split("(")[0].strip()
            if head:
                roots.append(head.lower())
        return roots

    def _references_other_agent(self, message: str, agent_names: Iterable[str]) -> bool:
        lowered = message.lower()
        for root in self._display_name_roots(agent_names):
            if root and re.search(rf"\b{re.escape(root)}\b", lowered):
                return True
        return False

    def _needs_rewrite(
        self,
        decision: AgentDecision,
        recent_contents: list[str],
        recent_speakers: list[str],
        *,
        discussion_mode: str,
        objective_tier: str,
        is_instigator: bool,
    ) -> tuple[bool, str]:
        if decision.action != "speak":
            return False, ""

        min_words, max_words = self._word_bounds_for_mode(discussion_mode, objective_tier)
        words = self._word_count(decision.message)
        if words < min_words:
            return True, f"too short ({words} words)"
        if words > max_words:
            return True, f"too long ({words} words)"

        if self._is_low_substance_message(decision.message):
            return True, "low substance or question/agreement loop"

        if self._is_botty_tone(decision.message):
            return True, "too formal/robotic tone"

        similarity = self._max_similarity_to_recent(decision.message, recent_contents)
        if similarity >= self.settings.duplicate_similarity_threshold:
            return True, f"too similar to recent messages (similarity={similarity:.2f})"

        if discussion_mode != "objective" and self.settings.attack_intensity == "high" and recent_speakers:
            if not self._references_other_agent(decision.message, recent_speakers):
                return True, "missing direct engagement with another agent"

        if self.settings.enforce_taunt_profanity and discussion_mode == "adversarial":
            if recent_speakers and not self._references_other_agent(decision.message, recent_speakers):
                return True, "missing direct taunt target by name"
            if not self._has_profanity(decision.message):
                return True, "missing profanity"

        if discussion_mode == "objective":
            if self._has_excessive_hostility(decision.message):
                return True, "tone too hostile for objective mode"

            if is_instigator and self._is_pure_hostile_without_factual_content(decision.message):
                return True, "instigator must include factual content in objective mode"

            if objective_tier == "simple" and not self._is_correction_message(decision.message):
                if self._is_redundant_objective_answer(decision.message, recent_contents):
                    return True, "redundant simple-objective answer"

        if discussion_mode == "exploratory":
            if self._has_excessive_hostility(decision.message):
                return True, "too hostile for exploratory mode"
            if self._is_overly_generic_exploratory(decision.message):
                return True, "exploratory answer is too generic"

        if discussion_mode == "adversarial" and is_instigator and not self._has_aggressive_tone(decision.message):
            return True, "instigator tone is too tame"

        return False, ""

    @staticmethod
    def _is_botty_tone(message: str) -> bool:
        lowered = message.lower()
        robotic_markers = (
            "i strongly disagree",
            "i firmly believe",
            "one concrete reason",
            "furthermore",
            "in conclusion",
            "moreover",
            "i propose we",
            "i reaffirm",
            "it is crucial",
        )
        if any(marker in lowered for marker in robotic_markers):
            return True

        # Overly rigid sentence starts often look synthetic in chat.
        starts = re.findall(r"(?:^|[.!?]\s+)([A-Z][^.!?]{0,80})", message)
        rigid_count = 0
        for chunk in starts:
            c = chunk.strip().lower()
            if c.startswith(("i believe", "i think", "i disagree", "i agree", "therefore", "however")):
                rigid_count += 1
        return rigid_count >= 3

    @staticmethod
    def _has_aggressive_tone(message: str) -> bool:
        lowered = message.lower()
        profanity = {
            "stfu",
            "fuck",
            "fucking",
            "fuck off",
            "bullshit",
            "shit",
            "dumb",
            "idiotic",
            "nonsense",
        }
        if any(token in lowered for token in profanity):
            return True

        attack_markers = (
            "you are wrong",
            "that's wrong",
            "your argument fails",
            "this is weak",
            "you're dodging",
            "stop dodging",
            "that's lazy reasoning",
            "that claim collapses",
        )
        return any(marker in lowered for marker in attack_markers)

    @staticmethod
    def _has_profanity(message: str) -> bool:
        lowered = message.lower()
        profanity_terms = (
            "fuck",
            "fucking",
            "fuck off",
            "stfu",
            "shit",
            "bullshit",
            "dumbass",
            "idiot",
            "moron",
            "asshole",
        )
        return any(term in lowered for term in profanity_terms)

    @staticmethod
    def _has_excessive_hostility(message: str) -> bool:
        lowered = message.lower()
        hard_hostility = (
            "fuck off",
            "stfu",
            "you are an idiot",
            "you are dumb",
            "moron",
            "asshole",
        )
        return any(token in lowered for token in hard_hostility)

    def _is_pure_hostile_without_factual_content(self, message: str) -> bool:
        lowered = message.lower()
        has_hostility = self._has_aggressive_tone(message) or self._has_profanity(message)
        factual_markers = (
            "because",
            "for example",
            "defined as",
            "evidence",
            "method",
            "data",
            "step",
            "means",
            "is",
        )
        has_factual = any(marker in lowered for marker in factual_markers)
        return has_hostility and not has_factual

    def _word_bounds_for_mode(self, discussion_mode: str, objective_tier: str) -> tuple[int, int]:
        if discussion_mode == "objective":
            if objective_tier == "simple":
                return self.settings.objective_simple_min_words, self.settings.objective_simple_max_words
            return self.settings.objective_min_words, self.settings.objective_max_words
        if discussion_mode == "exploratory":
            return self.settings.debatable_min_words, self.settings.debatable_max_words
        if discussion_mode == "adversarial":
            return self.settings.adversarial_min_words, self.settings.adversarial_max_words
        return self.settings.debatable_min_words, self.settings.debatable_max_words

    @staticmethod
    def _is_overly_generic_exploratory(message: str) -> bool:
        lowered = message.lower()
        generic_markers = (
            "it depends",
            "hard to say",
            "complex question",
            "can't really",
            "as an ai",
            "difficult to answer",
        )
        if any(marker in lowered for marker in generic_markers):
            return True

        # Exploratory answers should include some vivid descriptor.
        vivid_markers = (
            "like",
            "feels",
            "texture",
            "color",
            "rhythm",
            "tone",
            "edge",
            "shape",
        )
        return not any(marker in lowered for marker in vivid_markers)

    def _is_redundant_objective_answer(self, message: str, recent_contents: list[str]) -> bool:
        if not recent_contents:
            return False
        similarity = self._max_similarity_to_recent(message, recent_contents[-6:])
        return similarity >= 0.45

    @staticmethod
    def _is_correction_message(message: str) -> bool:
        lowered = message.lower()
        markers = (
            "wrong",
            "incorrect",
            "not true",
            "that's false",
            "correction",
            "actually",
            "to be precise",
        )
        return any(marker in lowered for marker in markers)

    def _objective_correction_count(self, recent_contents: list[str]) -> int:
        return sum(1 for content in recent_contents if self._is_correction_message(content))

    @staticmethod
    def _extract_discussion_mode(messages: list[dict[str, Any]]) -> str:
        if not messages:
            return "debatable"
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return "debatable"
        content = str(user_msgs[-1].get("content", ""))
        match = re.search(r"^Discussion Mode:\s*(objective|debatable|adversarial|exploratory)\s*$", content, flags=re.MULTILINE)
        if match:
            return match.group(1).strip().lower()
        return "debatable"

    @staticmethod
    def _extract_objective_tier(messages: list[dict[str, Any]]) -> str:
        if not messages:
            return "none"
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return "none"
        content = str(user_msgs[-1].get("content", ""))
        match = re.search(r"^Objective Tier:\s*(simple|explainer|none)\s*$", content, flags=re.MULTILINE)
        if match:
            return match.group(1).strip().lower()
        return "none"

    @staticmethod
    def _extract_recent_agent_context(messages: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        if not messages:
            return [], []
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if not user_msgs:
            return [], []
        content = str(user_msgs[-1].get("content", ""))

        speakers = re.findall(r"^speaker=(.+)$", content, flags=re.MULTILINE)
        bodies = re.findall(r"^content=(.+)$", content, flags=re.MULTILINE)

        filtered_speakers = [s.strip() for s in speakers if s.strip() and s.strip().lower() != "unknown"]
        filtered_bodies = [b.strip() for b in bodies if b.strip()]
        return filtered_bodies[-12:], filtered_speakers[-12:]

    async def summarize(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        *,
        timeout_sec: int,
    ) -> str:
        result = await self._complete_with_retries(
            provider=provider,
            model=model,
            messages=messages,
            timeout_sec=timeout_sec,
            max_tokens=self.settings.agent_summarize_max_tokens,
        )
        return result.text

    async def _complete_with_retries(
        self,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        timeout_sec: int,
        max_tokens: int,
    ) -> CompletionResult:
        attempts = max(1, self.settings.provider_retry_max_attempts)
        base_backoff = max(1, self.settings.provider_backoff_base_seconds)
        
        current_messages = list(messages)
        tools = get_tool_schema()
        
        # Track tools usage across multiple step calls
        tools_used: dict[str, int] = {}
        tool_results_texts = []
        total_latency_ms = 0.0
        consecutive_no_result_steps = 0
        
        # Max steps for tool calling loop
        for step in range(4):
            for attempt in range(attempts):
                await self.rate_limits.acquire(provider)
                try:
                    started = time.perf_counter()
                    response = await asyncio.wait_for(
                        acompletion(
                            model=model,
                            messages=current_messages,
                            temperature=self._completion_temperature(provider),
                            max_tokens=max_tokens,
                            tools=tools,
                            **self._provider_kwargs(provider),
                        ),
                        timeout=timeout_sec,
                    )
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    total_latency_ms += latency_ms
                    
                    choice = response.choices[0]
                    message = choice.message
                    
                    if choice.finish_reason == "tool_calls" or hasattr(message, "tool_calls") and message.tool_calls:
                        # Append the assistant's request
                        current_messages.append(message.model_dump())
                        step_no_result_only = True
                        
                        for tool_call in message.tool_calls:
                            func = tool_call.function
                            tool_name = func.name
                            tools_used[tool_name] = tools_used.get(tool_name, 0) + 1
                            
                            # Execute tool
                            tool_result = await dispatch_tool(tool_name, func.arguments)
                            if not str(tool_result).startswith("WEB_SEARCH_NO_RESULTS"):
                                step_no_result_only = False
                            
                            # Record context for the final output embed presentation
                            arg_str = str(func.arguments)
                            if len(arg_str) > 50: arg_str = arg_str[:47] + "..."
                            res_str = str(tool_result)
                            if len(res_str) > 100: res_str = res_str[:97] + "..."
                            tool_results_texts.append(f"> 🛠️ **Tool Used:** {tool_name}({arg_str})\n> Result: {res_str}\n")
                            
                            # Append tool response back to the LLM
                            current_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": tool_result
                            })

                        if step_no_result_only:
                            consecutive_no_result_steps += 1
                        else:
                            consecutive_no_result_steps = 0

                        if consecutive_no_result_steps >= 2:
                            return CompletionResult(
                                text='{"action":"pass","message":"Skipping this turn because web search produced no usable results."}',
                                latency_ms=total_latency_ms,
                                attempts_used=attempt + 1,
                                tools_used=tools_used,
                                tool_results_text="\n".join(tool_results_texts) if tool_results_texts else None,
                            )
                        
                        # Break retry loop to go to next step
                        break
                        
                    # Final textual response
                    return CompletionResult(
                        text=(message.content or "").strip(),
                        latency_ms=total_latency_ms,
                        attempts_used=attempt + 1,
                        tools_used=tools_used,
                        tool_results_text="\n".join(tool_results_texts) if tool_results_texts else None,
                    )
                except asyncio.TimeoutError:
                    if attempt == attempts - 1:
                        raise
                    await asyncio.sleep(base_backoff * (2**attempt))
                except Exception as exc:  # pragma: no cover - provider-specific error surfaces vary.
                    if self.is_fatal_model_error(exc):
                        # Do not retry fatal errors like NotFoundError 
                        raise
                        
                    if self._is_rate_limited(exc):
                        retry_after = self._extract_retry_after(exc)
                        wait_time = retry_after if retry_after is not None else float(base_backoff * (2**attempt))
                        wait_time = min(self.MAX_RETRY_AFTER_SECONDS, max(0.5, wait_time))
                        await self.rate_limits.apply_retry_after(provider, wait_time)
                        if attempt == attempts - 1:
                            raise
                        logger.warning("Rate limit from %s. Retrying step %s in %.2fs", provider, step, wait_time)
                        continue

                    if attempt == attempts - 1:
                        raise
                    await asyncio.sleep(base_backoff * (2**attempt))

        raise RuntimeError("Unexpected retry loop termination or max tool steps exceeded")

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True

        raw = str(exc).lower()
        return "429" in raw or "rate limit" in raw

    @staticmethod
    def is_fatal_model_error(exc: Exception) -> bool:
        if exc is None:
            return False

        name = type(exc).__name__.lower()
        if any(token in name for token in ("notfound", "not_found", "resource_not_found", "unsupported", "invalid", "badrequest", "bad_request")):
            return True

        if getattr(exc, "status_code", None) == 404:
            return True

        raw = str(exc).lower()
        patterns = (
            "not found", 
            "404", 
            "model not found", 
            "resource not found", 
            "not supported", 
            "unsupported", 
            "invalid model", 
            "does not support", 
            "bad request",
            "audio model",
            "transcription",
            "text-to-speech",
            "chat/completions",
            "not a completion",
            "only available for",
        )
        if any(token in raw for token in patterns):
            return True

        return False

    @staticmethod
    def _completion_temperature(provider: str) -> float:
        key = provider.lower().strip()
        if key in {"google", "gemini", "google_ai_studio", "google-ai-studio", "googleai"}:
            return 1.0
        return 0.4

    @staticmethod
    def _extract_retry_after(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            for key in ("retry-after", "Retry-After"):
                value = headers.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue

        body = str(exc)
        for token in body.replace("=", " ").split():
            token = token.strip().strip(",")
            try:
                value = float(token)
            except ValueError:
                continue
            if 0 < value <= LLMRouter.MAX_RETRY_AFTER_SECONDS:
                return value
        return None

    @staticmethod
    def _extract_json_blocks(text: str) -> list[str]:
        """Extract all top-level JSON objects from text using brace matching."""
        cleaned = text.strip()
        cleaned = LLMRouter._strip_outer_speak_wrapper(cleaned)

        # Remove markdown fences (any variant: ```json, ```JSON, ```)
        cleaned = re.sub(r"```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        results: list[str] = []
        depth = 0
        start = None
        for i, char in enumerate(cleaned):
            if char == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = cleaned[start:i+1]
                    results.append(candidate)
                    start = None
        return results

    @staticmethod
    def _validate_decision_block(block: dict) -> tuple[bool, str]:
        """Validate a parsed JSON block against the decision schema.
        Returns (valid, error_message)."""
        if not isinstance(block, dict):
            return False, "not a JSON object"

        action = block.get("action")
        if action is None:
            return False, "missing 'action'"
        if not isinstance(action, str) or action.lower().strip() not in {"speak", "pass"}:
            return False, f"invalid action: {action}"

        message = block.get("message")
        if message is None:
            return False, "missing 'message'"
        if not isinstance(message, str):
            return False, "'message' is not a string"

        return True, ""

    @staticmethod
    def _block_to_decision(
        block: dict,
        raw_text: str,
        latency_ms: float,
        provider: str,
        model: str,
        attempts_used: int,
    ) -> AgentDecision:
        """Convert a validated JSON block into an AgentDecision."""
        action = str(block.get("action", "speak")).lower().strip()
        message = str(block.get("message", "")).strip()
        quote_id = block.get("quote_message_id")
        if quote_id is not None:
            quote_id = str(quote_id).strip()
            if not quote_id.isdigit():
                quote_id = None

        if not message:
            message = "I have no additional points."

        return AgentDecision(
            action=action,
            message=LLMRouter._sanitize_message(message),
            raw_text=raw_text,
            latency_ms=latency_ms,
            provider=provider,
            model=model,
            attempts_used=attempts_used,
            quote_message_id=quote_id,
        )

    @staticmethod
    def _parse_decision(
        raw_text: str,
        *,
        latency_ms: float,
        provider: str,
        model: str,
        attempts_used: int,
    ) -> AgentDecision:
        json_blocks = LLMRouter._extract_json_blocks(raw_text)

        if not json_blocks:
            # Fallback: try regex extraction of message field
            pattern = r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"'
            match = re.search(pattern, raw_text)
            if match:
                extracted_msg = match.group(1)
                try:
                    extracted_msg = extracted_msg.encode("utf-8").decode("unicode_escape")
                except UnicodeDecodeError:
                    extracted_msg = match.group(1)
                return AgentDecision(
                    action="speak",
                    message=LLMRouter._sanitize_message(extracted_msg),
                    raw_text=raw_text,
                    latency_ms=latency_ms,
                    provider=provider,
                    model=model,
                    attempts_used=attempts_used,
                )

            logger.warning(
                "Discarding malformed model payload from %s/%s. Raw preview: %s",
                provider,
                model,
                raw_text[:200].replace("\n", " "),
            )
            return AgentDecision(
                action="pass",
                message="Skipping this turn due to a response formatting issue.",
                raw_text=raw_text,
                latency_ms=latency_ms,
                provider=provider,
                model=model,
                attempts_used=attempts_used,
            )

        # Parse and validate each block
        valid_decisions: list[AgentDecision] = []
        for block_str in json_blocks:
            try:
                block = json.loads(block_str)
            except json.JSONDecodeError:
                continue

            is_valid, error = LLMRouter._validate_decision_block(block)
            if not is_valid:
                logger.debug("Skipping invalid decision block: %s", error)
                continue

            decision = LLMRouter._block_to_decision(
                block, raw_text, latency_ms, provider, model, attempts_used,
            )
            valid_decisions.append(decision)

        if not valid_decisions:
            # All blocks were invalid, fall back to pass
            return AgentDecision(
                action="pass",
                message="Skipping this turn due to invalid decision format.",
                raw_text=raw_text,
                latency_ms=latency_ms,
                provider=provider,
                model=model,
                attempts_used=attempts_used,
            )

        # First decision is the primary, rest are multi-messages
        primary = valid_decisions[0]
        if len(valid_decisions) > 1:
            primary = AgentDecision(
                action=primary.action,
                message=primary.message,
                raw_text=raw_text,
                latency_ms=latency_ms,
                provider=provider,
                model=model,
                attempts_used=attempts_used,
                quote_message_id=primary.quote_message_id,
                multi_messages=valid_decisions[1:],
            )

        return primary

    @staticmethod
    def _strip_outer_speak_wrapper(text: str) -> str:
        candidate = text.strip()
        lowered = candidate.lower()
        if lowered.startswith("<speak>") and lowered.endswith("</speak>"):
            return candidate[len("<speak>") : -len("</speak>")].strip()
        return candidate

    @staticmethod
    def _sanitize_message(text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return cleaned

        # Recover from serialized wrapper payloads before any final cleanup.
        parsed = None
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                wrapped_message = parsed.get("message")
                if wrapped_message is not None:
                    cleaned = str(wrapped_message)
                else:
                    cleaned = ""

        if not cleaned and text:
            msg_match = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if msg_match:
                cleaned = msg_match.group(1)

        cleaned = re.sub(r"<\s*/?speak[^>]*>", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()

        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`").strip()

        # Strip leaked JSON fields from message content (quote_message_id, action, etc.)
        cleaned = re.sub(r"quote_message_id\s*=?\s*\d+", "", cleaned).strip()
        cleaned = re.sub(r'"quote_message_id"\s*:\s*"\d+"', "", cleaned).strip()
        cleaned = re.sub(r'"action"\s*:\s*"(speak|pass)"', "", cleaned).strip()

        # Hard cap: truncate to 2500 chars to allow thoughtful, detailed responses
        if len(cleaned) > 2500:
            cleaned = cleaned[:2497].rstrip() + "..."

        return cleaned

    def _provider_kwargs(self, provider: str) -> dict[str, str]:
        key = provider.lower().strip()
        if key == "groq":
            return self._compact_kwargs(self.settings.groq_api_key, self.settings.groq_api_base)
        if key == "mistral":
            return self._compact_kwargs(self.settings.mistral_api_key, self.settings.mistral_api_base)
        if key == "cerebras":
            return self._compact_kwargs(self.settings.cerebras_api_key, self.settings.cerebras_api_base)
        if key == "sambanova":
            return self._compact_kwargs(self.settings.sambanova_api_key, self.settings.sambanova_api_base)
        if key in {"google", "gemini", "google_ai_studio", "google-ai-studio", "googleai"}:
            return self._compact_kwargs(self.settings.google_api_key, self.settings.google_api_base)
        return {}

    @staticmethod
    def _compact_kwargs(api_key: str, api_base: str) -> dict[str, str]:
        payload: dict[str, str] = {}
        if api_key:
            payload["api_key"] = api_key
        if api_base:
            payload["api_base"] = api_base
        return payload
