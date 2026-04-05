Discord Council Orchestrator
============================

Multi-agent debate orchestration in Discord with live human steering.

Quick start
-----------

1. pip install -e .
2. Copy .env.example to .env and fill API keys.
3. python -m src.app.main

Usage
-----

/council <topic>       -- start a session
/models <provider>     -- list models (sambanova, google, all)
<any message>          -- steer the council while active

Config
------

Agent personas: src/config/agents.yaml (must define at least 5)
Settings:       src/config/settings.py

Providers: GROQ_API_KEY, MISTRAL_API_KEY, CEREBRAS_API_KEY, SAMBANOVA_API_KEY, GOOGLE_API_KEY

Notes
-----

Bot needs Manage Webhooks permission and message content intent enabled.

License
-------

Good Faith License (GFL) v1.0. See LICENSE.
