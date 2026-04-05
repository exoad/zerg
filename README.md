# Discord Council Orchestrator

Multi-agent debate orchestration in Discord with live human steering.

## What this does

- Runs one Discord bot as the control plane.
- Uses webhooks to present multiple agent personas in one channel.
- Lets a human start a session, then send follow-up steer messages at any time.
- Keeps all agents in shared context so they can react to each other.
- Runs 5 active debate agents plus a dedicated moderator profile.
- Reshuffles active agent provider/model assignments at session start and after each accepted human steer.
- Enforces turn caps and provider rate-limit guardrails for free tiers.
- Sends moderator finalization as embeds for high visibility.
- Attaches lightweight debug embeds to agent messages (latency/provider/model).
- Supports multi-provider councils via Groq, Mistral, Cerebras, SambaNova, and Google AI Studio (Gemini).
- Default agent assignment strategy excludes Google from active debate turns because its free-tier limits are too restrictive for council pacing.

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -e .`
3. Copy `.env.example` to `.env` and fill API keys.
4. Run:
   - `python -m src.app.main`

## Channel lock

- The bot is hard-locked to a single Discord channel for all reads and writes.
- Configure `DISCORD_ALLOWED_CHANNEL_ID` (default: `1490122012574748814`).
- Commands and steer messages outside that channel are ignored.
- Moderator/Zerg control-plane responses are delivered as embeds for visibility.

## Provider setup

- Fill any provider keys you plan to use in `.env`:
   - `GROQ_API_KEY`
   - `MISTRAL_API_KEY`
   - `CEREBRAS_API_KEY`
   - `SAMBANOVA_API_KEY`
   - `GOOGLE_API_KEY`
- You can choose provider per agent in `src/config/agents.yaml` using model prefixes like:
   - `groq/...`
   - `mistral/...`
   - `cerebras/...`
   - `sambanova/...`
   - `gemini/...`

- Free-tier-safe defaults are preconfigured in `.env.example` with conservative RPM budgets
  to reduce 429s across providers.

## Model discovery command

- Fetch SambaNova models:
   - `/models sambanova`
- Fetch Google AI Studio Gemini models:
   - `/models google`
- Fetch all configured provider model lists:
   - `/models all`

## Discord usage

- Start a council session:
   - `/council <topic>`
- While active, any normal (non-slash) message from the starter user in the allowed channel is treated as a steer message.

## Consensus pacing and debug

- `COUNCIL_MAX_TURNS`: hard cap on total turns (default 12).
- `CONSENSUS_THRESHOLD`: lexical similarity needed for consensus (default 0.82).
- `CONSENSUS_MIN_RECENT_MESSAGES`: number of recent non-stale messages considered before consensus is allowed (default 5).
- `CONSENSUS_MIN_UNIQUE_AGENTS`: minimum distinct agents that must have spoken before consensus (default 3).
- `CONSENSUS_MIN_TURNS`: minimum turn index before consensus can close the session (default 6).
- `CONSENSUS_PASS_COUNT_REQUIRED`: pass-only shortcut threshold. Set `0` to disable fast-close from pass events.
- `AGENT_DEBUG_EMBEDS_ENABLED`: attach per-message debug embeds with latency/provider/model/attempts.

Final consensus output is posted as embeds and includes session debug details:
- total duration
- total turns
- participating agents
- per-agent performance rollup
- provider activity

## Notes

- Bot needs `Manage Webhooks` permission in target channels.
- Bot also needs message content intent enabled.
- Agent personas are in `src/config/agents.yaml`.
- `src/config/agents.yaml` must define at least 5 active agent profiles.
- The runtime uses LiteLLM for chat completions and OpenAI-compatible provider `/models` APIs for model discovery.

## Runtime assignment policy

- Active agent assignments are refreshed when a session starts.
- Assignments are refreshed again after each accepted human steer message.
- Each refresh tries to keep model IDs unique across active agents.
- Providers can repeat across agents.
- If model discovery is insufficient, configured agent defaults are used as fallback.

## Reply attribution format

- If an agent responds to a specific message ID, output is prefixed with a directional label:
   - `[Reply: <agent> -> <target> | msg_id=<id>]`
- If target resolution fails, output is prefixed with:
   - `[Reply: <agent> -> msg_id=<id> | unresolved]`
- Debug embeds include an `In Response To` field with target actor, target message id, and resolution status.
