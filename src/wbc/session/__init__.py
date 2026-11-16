"""The session layer: one Session per robot connection, its host process and its HTTP client.

- ``session``: the state machine (``Session``, ``OPS``); ``runs``: appended plans and their progress; ``store``: the
  session directory; ``observation``, ``perception``, ``views``, ``depth_refine``: what an observation holds and how
  perceive measures it.
- ``host``: the session host (supervisor and worker, ``python -m wbc.session.host``); ``client``: its HTTP client;
  ``cli``: the ``wbc`` command line on the client.

Nothing is imported here: the agent layer imports ``client`` alone, without the planner or MuJoCo.
"""
