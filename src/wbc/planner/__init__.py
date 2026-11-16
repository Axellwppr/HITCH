"""From a request to a checked motion that continues the reference tail.

- ``request``: the agent's structured request (docs/REFERENCE.md); ``plan`` is the session's only planner.
- ``hands``: what each hand does over the in-place body; ``arms``: the arms solved over it, on one clock.
- ``steps``: the motion skills of a request (its walk and its stand-still) run from a state, arms at their joint
  angles, and the checks and padding every plan finishes with.
"""

from .request import plan
from .steps import Planned

__all__ = ["Planned", "plan"]
