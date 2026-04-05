from __future__ import annotations

import asyncio
import json
import logging
import re
import time
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
    reply_to_message_id: str | None = None
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
        dec = self._parse_decision(
            result.text,
            latency_ms=result.latency_ms,
            provider=actual_provider,
            model=actual_model,
            attempts_used=result.attempts_used,
        )
        return AgentDecision(
            action=dec.action,
            message=dec.message,
            raw_text=dec.raw_text,
            latency_ms=dec.latency_ms,
            provider=dec.provider,
            model=dec.model,
            attempts_used=dec.attempts_used,
            reply_to_message_id=dec.reply_to_message_id,
            tools_used=result.tools_used,
            tool_results=result.tool_results_text,
        )

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

        reply_to = block.get("reply_to_message_id")
        if reply_to is not None:
            reply_str = str(reply_to).strip()
            if not reply_str.isdigit():
                return False, f"reply_to_message_id is not numeric: {reply_str}"

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
        reply_to = block.get("reply_to_message_id")
        if reply_to is not None:
            reply_to = str(reply_to).strip()
            if not reply_to.isdigit():
                reply_to = None

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
            reply_to_message_id=reply_to,
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
                reply_to_message_id=primary.reply_to_message_id,
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

        # Hard cap: truncate to 400 chars to enforce Discord-style brevity
        if len(cleaned) > 400:
            cleaned = cleaned[:397].rstrip() + "..."

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
