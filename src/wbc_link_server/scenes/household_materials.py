"""The household albedo textures (assets/materials/household); material sampling leaves physical geometry untouched."""

import mujoco
import numpy as np

from wbc.paths import ASSETS

MATERIALS = ASSETS / "materials" / "household"
FLOORS = ("floor_mint_terrazzo", "floor_terracotta")
WOODS = ("furniture_cobalt", "furniture_coral")


def sample(seed):
    """Choose independent albedos for the floor and both supports without changing geometry."""
    rng = np.random.default_rng([seed, 1771])
    return {
        "floor": str(rng.choice(FLOORS)),
        "source": str(rng.choice(WOODS)),
        "destination": str(rng.choice(WOODS)),
    }


def apply(spec, layout):
    """Texture the floor and the furniture boards (planar, metric UVs). Objects keep
    their own colours."""
    selected = layout["appearance"]["materials"]
    added = {}

    def material(asset, tile_m):
        key = (asset, tile_m)
        if key not in added:
            name = f"household_{asset}_2d"
            spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(MATERIALS / (asset + ".png")))
            mat = spec.add_material(
                name=name, texuniform=True, texrepeat=[1 / tile_m, 1 / tile_m], specular=0.08, shininess=0.12
            )
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
            added[key] = name
        return added[key]

    floor = spec.geom("floor")
    floor.material = material(selected["floor"], 2.0)
    floor.rgba = [1, 1, 1, 1]
    for furniture in layout["furniture"]:
        role = furniture["id"]
        top_material = material(selected[role], 1.2)
        for name in [s["geom"] for s in furniture["surfaces"]]:
            geom = spec.geom(name)
            geom.material = top_material
            geom.rgba = [1, 1, 1, 1]
