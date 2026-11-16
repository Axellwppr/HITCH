"""G1 whole-body harness: observe, perceive, plan a request from the reference tail, append, track.

The top-level names are imported lazily so that light modules (``wbc.core``,
``wbc.link``) stay usable on the robot computer without MuJoCo or rendering.
"""

__all__ = ["Session"]


def __getattr__(name):
    if name in __all__:
        from wbc.session import session

        return getattr(session, name)
    raise AttributeError(name)
