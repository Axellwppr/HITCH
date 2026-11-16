"""Small MuJoCo primitives for the simulator scenes."""

import mujoco


def box(parent, name, center, half_size, rgba):
    return parent.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=center,
        size=half_size,
        rgba=rgba,
        quat=[1, 0, 0, 0],
        friction=[0.6, 0.005, 0.0001],
    )
