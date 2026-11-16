"""An agent loop over the session host: a model sees observations and calls the harness as tools.

The loop owns the whole context (``prompts/`` and ``tasks/`` markdown, the
append-only conversation), talks to any OpenAI-compatible model server
(``backends``), and to the robot only through the session host's HTTP API.
"""
