"""Single-hand pick and place: put one small object into a container or onto a mat (Dex3 hands, grip).

One to three objects stand on a table, counter or round table (the source); the task names one of them and a
destination: an open bin or bucket, or a flat mat, on the same top or on a second piece of furniture nearby
(placed as household layouts are). The objects are cubes, balls, upright boxes and cylinders, in distinct colours.

The robot starts either at the source, the objects within a step's reach, or 0.5-1.2 m back from it and turned up
to 60 degrees away. Built by the household builder (furniture, room, materials) with the destination as a static
fixture; objects keep their plain colours.
"""

from __future__ import annotations

import numpy as np

from wbc_link_server.scenes import household, household_materials
from wbc_link_server.scenes.household import NAMES as FURNITURE_NAMES
from wbc_link_server.scenes.household import (
    _coarse_heading,
    _destination_pose,
    _direction,
    _furniture,
    rotation,
)

KIND = "tabletop_pick"
build = household.build  # the household builder (furniture, room, materials; the destination a static fixture)
TOP_RANGE = (0.40, 0.90)  # m, the height of the furniture tops
PICK_ONLY = 0.25  # share of scenes whose task is only to pick the object up and hold it
SOURCE_KINDS = ("table", "round_table", "counter")
DESTINATION_KINDS = ("table", "round_table", "counter", "stool")
COLORS = {
    "red": [0.85, 0.10, 0.10, 1],
    "blue": [0.10, 0.30, 0.85, 1],
    "green": [0.10, 0.60, 0.20, 1],
    "yellow": [0.95, 0.80, 0.10, 1],
    "purple": [0.50, 0.20, 0.70, 1],
    "orange": [0.90, 0.33, 0.02, 1],
    "cyan": [0.05, 0.70, 0.75, 1],
    "pink": [0.95, 0.45, 0.65, 1],
    "black": [0.08, 0.08, 0.09, 1],
}
# The object shapes; every object stands upright and is graspable by one Dex3 hand.
NOUNS = {"cube": "cube", "sphere": "ball", "box": "box", "cylinder": "cylinder"}
FAMILIES = tuple(NOUNS)
SQUARE = np.radians(10)  # objects stand within this of square to their furniture's front
WALL = 0.01  # container walls and floor, m
# Sliding, torsional and rolling friction of balls: fruit-like, a ball rolls a little and comes to rest.
ROUND_FRICTION = (0.8, 0.02, 0.01)
# m behind a top's near edge where objects and destinations stand, their centres at most the second: within reach of
# a robot standing at the edge.
FRONT_BAND = (0.07, 0.20)
# Clearance around the named object, in its furniture's frame (x inward from the front, y left), that the other
# objects keep: room for a right hand coming in from the side or from above.
HAND_ROOM = ((-0.09, 0.16), (-0.24, 0.06))


def _item_shape(rng, family):
    """MuJoCo type, size, height and footprint half extents (x, y) of an object of ``family``."""
    # Sizes of hand-held objects: cubes and balls 7.4-9 cm.
    if family == "cube":
        edge = rng.uniform(0.074, 0.09)
        return "box", [edge / 2] * 3, edge, (edge / 2, edge / 2)
    if family == "sphere":
        r = rng.uniform(0.037, 0.045)
        return "sphere", [r], 2 * r, (r, r)
    if family == "box":
        width, depth, height = rng.uniform(0.04, 0.06), rng.uniform(0.04, 0.07), rng.uniform(0.09, 0.16)
        return "box", [depth / 2, width / 2, height / 2], height, (depth / 2, width / 2)
    r, height = rng.uniform(0.025, 0.038), rng.uniform(0.09, 0.18)  # a cylinder
    return "cylinder", [r, height / 2], height, (r, r)


def _destination(rng):
    """A container or mat as a fixture at the origin: geoms, its footprint half extents and the facts to score it."""
    kind = str(rng.choice(("bin", "bucket", "mat", "disk")))
    if kind == "bin":
        inner = np.array([rng.uniform(0.15, 0.22), rng.uniform(0.17, 0.26)]) / 2
        height = rng.uniform(0.05, 0.09)
        hx, hy = inner + WALL
        geoms = [{"name": "floor", "type": "box", "pos": [0, 0, WALL / 2], "size": [hx, hy, WALL / 2]}]
        for i, (x, y, sx, sy) in enumerate(
            (
                (hx - WALL / 2, 0, WALL / 2, hy),
                (-hx + WALL / 2, 0, WALL / 2, hy),
                (0, hy - WALL / 2, hx, WALL / 2),
                (0, -hy + WALL / 2, hx, WALL / 2),
            )
        ):
            geoms.append({"name": f"wall_{i}", "type": "box", "pos": [x, y, height / 2], "size": [sx, sy, height / 2]})
        facts = {"type": "bin", "inner_half": inner.tolist(), "floor_z": WALL, "rim_z": height}
        return kind, geoms, (hx, hy), facts
    if kind == "bucket":
        radius, height = rng.uniform(0.08, 0.11), rng.uniform(0.06, 0.10)
        geoms = [{"name": "floor", "type": "cylinder", "pos": [0, 0, WALL / 2], "size": [radius + WALL, WALL / 2]}]
        segments = 20
        chord = 2 * (radius + WALL) * np.tan(np.pi / segments) + 0.002
        for i in range(segments):
            a = 2 * np.pi * i / segments
            geoms.append(
                {
                    "name": f"wall_{i}",
                    "type": "box",
                    "pos": [(radius + WALL / 2) * np.cos(a), (radius + WALL / 2) * np.sin(a), height / 2],
                    "size": [WALL / 2, chord / 2, height / 2],
                    "yaw": a,
                }
            )
        facts = {"type": "bucket", "inner_radius": radius, "floor_z": WALL, "rim_z": height}
        return kind, geoms, (radius + WALL, radius + WALL), facts
    thickness = 0.004
    if kind == "mat":
        half = np.array([rng.uniform(0.07, 0.11), rng.uniform(0.07, 0.11)])
        geoms = [{"name": "top", "type": "box", "pos": [0, 0, thickness / 2], "size": [*half, thickness / 2]}]
        return kind, geoms, tuple(half), {"type": "mat", "half": half.tolist(), "floor_z": thickness}
    radius = rng.uniform(0.08, 0.11)
    geoms = [{"name": "top", "type": "cylinder", "pos": [0, 0, thickness / 2], "size": [radius, thickness / 2]}]
    return kind, geoms, (radius, radius), {"type": "disk", "radius": radius, "floor_z": thickness}


DESTINATION_NOUNS = {"bin": "bin", "bucket": "bucket", "mat": "mat", "disk": "round mat"}


def _top_height(furniture, height):
    furniture["height"] = float(height)
    for surface in furniture["surfaces"]:
        surface["height"] = float(height)


def _front_spot(rng, furniture, half_xy, taken, clearance=0.06, keep_out=None):
    """A point in the front band of a furniture top (its local -x is the front) that keeps ``half_xy`` on the top,
    clear of the ``taken`` footprints (centre, half) and outside ``keep_out`` ((x lo, x hi), (y lo, y hi), in the
    furniture's frame); in world coordinates, or None."""
    surface = furniture["surfaces"][0]
    hx, hy = surface["half_xy"]
    for _ in range(60):
        depth = rng.uniform(FRONT_BAND[0] + half_xy[0], max(FRONT_BAND[0] + half_xy[0], FRONT_BAND[1]))
        local = np.array([-hx + depth, rng.uniform(-hy + 0.06 + half_xy[1], hy - 0.06 - half_xy[1])])
        if surface["shape"] == "disk" and np.linalg.norm(local) + max(half_xy) > hx - 0.03:
            continue
        if np.any(np.abs(local) + np.asarray(half_xy) > np.array([hx, hy]) - 0.02):
            continue
        if keep_out is not None:
            reach = np.linalg.norm(half_xy)
            (x0, x1), (y0, y1) = keep_out
            if x0 - reach < local[0] < x1 + reach and y0 - reach < local[1] < y1 + reach:
                continue
        world = np.asarray(furniture["center"]) + rotation(furniture["yaw"]) @ local
        reach = np.linalg.norm(half_xy)  # enclosing circles, so any small turn keeps them apart
        if all(np.linalg.norm(world - c) > reach + np.linalg.norm(h) + clearance for c, h in taken):
            return world
    return None


def _instruction(rng, goal, item, source, verb, destination, where):
    """The task in one of several equivalent wordings (``rng``: the scene's text stream)."""

    def pick(options):
        return str(options[int(rng.integers(len(options)))])

    if goal == "pick":
        return pick(
            (
                f"Pick up the {item} from the {source} with one hand and hold it.",
                f"Lift the {item} off the {source} using one hand.",
                f"Grab the {item} on the {source} and keep it in your hand.",
                f"Take the {item} from the {source} with one hand; you do not need to put it anywhere.",
                f"Use one hand to pick up the {item} from the {source}.",
                f"Please pick the {item} up off the {source} and hold it with one hand.",
            )
        )
    text = pick(
        (
            f"Put the {item} {verb} the {destination}{where}, using one hand.",
            f"Pick up the {item} with one hand and place it {verb} the {destination}{where}.",
            f"Move the {item} {verb} the {destination}{where} with one hand.",
            f"Using one hand, take the {item} and put it {verb} the {destination}{where}.",
            f"The {item} goes {verb} the {destination}{where}; carry it there in one hand.",
            f"Grab the {item} and put it {verb} the {destination}{where}; one hand is enough.",
            f"Please place the {item} {verb} the {destination}{where} with a single hand.",
            f"With one hand, bring the {item} over to the {destination}{where} and put it {verb} it.",
        )
    )
    return (
        text
        + " "
        + pick(
            (
                "Release it and check that it stays there.",
                "Then let go and make sure it stays there.",
                "Let go of it and check that it stays in place.",
            )
        )
    )


def make_scene(seed=0):
    rng = np.random.default_rng([seed, 1901])
    board_rng = np.random.default_rng([seed, 1911])  # the board thicknesses, their own stream
    text_rng = np.random.default_rng([seed, 1912])  # the instruction's wording
    goal = "pick" if np.random.default_rng([seed, 1913]).random() < PICK_ONLY else "place"
    source = _furniture(rng, "source", str(rng.choice(SOURCE_KINDS)))
    _top_height(source, rng.uniform(*TOP_RANGE))
    source["board"] = float(board_rng.uniform(*household.BOARD_RANGE))
    at_source = rng.random() < 0.4
    if at_source:  # the near edge 0.25-0.32 m ahead: the objects within a step
        source["center"] = [0.25 + float(rng.uniform(0, 0.07)) + source["surfaces"][0]["half_xy"][0], 0.0]
        source["yaw"] = float(rng.uniform(-0.08, 0.08))
        heading = 0.0
    else:
        source["center"] = [float(rng.uniform(0.85, 1.35)), float(rng.uniform(-0.35, 0.35))]
        source["yaw"] = float(rng.uniform(-np.pi / 12, np.pi / 12))
        heading = float(rng.uniform(-60, 60))
    colors = list(rng.permutation(list(COLORS)))
    count = int(rng.integers(1, 4))
    items, taken, keep_out = [], [], None
    for _ in range(count):
        family = str(rng.choice(FAMILIES))
        kind, size, height, half = _item_shape(rng, family)
        where = _front_spot(rng, source, half, taken, keep_out=keep_out)
        if where is None:
            continue
        taken.append((where, half))
        if keep_out is None:  # the named object: the hand's room around it stays free
            local = rotation(-source["yaw"]) @ (where - np.asarray(source["center"]))
            keep_out = tuple((local[k] + lo, local[k] + hi) for k, (lo, hi) in enumerate(HAND_ROOM))
        yaw = source["yaw"] + (rng.uniform(-SQUARE, SQUARE) if family in ("cube", "box") else 0.0)
        item = {
            "name": f"item_{len(items)}",
            "type": kind,
            "pos": [*map(float, where), source["height"] + height / 2 + 0.002],
            "size": [float(v) for v in size],
            "mass": float(rng.uniform(0.05, 0.30)),
            "dynamic": True,
            "rgba": COLORS[colors[len(items)]],
            "friction": [0.8, 0.005, 0.0001],
            "condim": 4,
            "yaw": float(yaw),
        }
        if kind == "sphere":  # fruit-like: it rolls a little, not like a billiard ball (condim 6: rolling friction)
            item.update(friction=list(ROUND_FRICTION), condim=6)
        items.append((item, family, height, half))
    target = items[0]
    destination_color = colors[len(items)]
    kind, geoms, footprint, facts = _destination(rng)
    furniture = [source]
    where = None
    if rng.random() < 0.5:  # on the same top, beside the objects
        where = _front_spot(rng, source, footprint, taken, clearance=0.10, keep_out=keep_out)
        place_on = source
    if where is None:
        place_on = _furniture(rng, "destination", str(rng.choice(DESTINATION_KINDS)))
        _top_height(place_on, rng.uniform(*TOP_RANGE))
        place_on["board"] = float(board_rng.uniform(*household.BOARD_RANGE))
        _destination_pose(rng, source, place_on)
        furniture.append(place_on)
        where = _front_spot(rng, place_on, footprint, [])
        if where is None:
            where = np.asarray(place_on["center"], float)
    fixture = {
        "name": "destination_fixture",
        "pos": [*map(float, where), place_on["height"]],
        "yaw": float(place_on["yaw"] + rng.uniform(-SQUARE, SQUARE)),
        "geoms": [{**g, "name": f"destination_{g['name']}", "rgba": COLORS[destination_color]} for g in geoms],
    }
    item_label = f"{colors[0]} {NOUNS[target[1]]}"
    destination_label = f"{destination_color} {DESTINATION_NOUNS[kind]}"
    hint = None  # the stated direction (degrees from the first view), as the instruction words it
    if place_on is source:
        where_text = f" on the same {FURNITURE_NAMES[source['kind']]}"
    else:
        bearing = float(np.degrees(np.arctan2(*np.asarray(place_on["center"])[::-1])))
        direction = _direction(bearing - heading, int(text_rng.integers(3)))
        where_text = f" on the {FURNITURE_NAMES[place_on['kind']]} {direction}"
        hint = _coarse_heading(bearing - heading)
    verb = "into" if kind in ("bin", "bucket") else "onto"
    instruction = _instruction(
        text_rng, goal, item_label, FURNITURE_NAMES[source["kind"]], verb, destination_label, where_text
    )
    objects = [item for item, *_ in items]
    materials = household_materials.sample(seed)
    return {
        "kind": KIND,
        "seed": seed,
        "objects": objects,
        "furniture": furniture,
        "fixtures": [fixture],
        "appearance": {"materials": materials},
        "settle_seconds": 0.3,
        "initial_offset": {"heading_deg": heading},
        "task": {"instruction": instruction},
        "task_spec": {
            "goal": goal,  # "pick": pick it up and hold it; "place": put it in or on the destination
            "item": "item_0",
            "item_family": target[1],
            "item_height": float(target[2]),
            "destination": facts
            | {"centre": fixture["pos"][:2], "yaw": fixture["yaw"], "base_z": fixture["pos"][2], "on": place_on["id"]},
            "destination_hint_deg": hint,
        },
    }
