"""Processes that run next to the controller: the link server, cameras, simulator.

``launch`` starts them (``python -m wbc_link_server.launch.sim``, ``.policy``, ``.control``), ``bridges`` is the
simulator behind the controller's protocol, ``sensors`` the simulated head camera, ``scenes`` our MuJoCo scenes.
"""
