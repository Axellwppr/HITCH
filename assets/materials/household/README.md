# Household albedo assets

A few generated tileable surface textures for the simulated room.

| Assets | Use |
|---|---|
| `floor_*` (2) | ground plane |
| `furniture_*` (2) | furniture tops and boards |

`household_materials.sample(seed)` draws a floor and one texture per piece of furniture for a scene.
The PNGs are 2048×2048 RGB. Each source image is mirrored into a 2×2 tile and resampled once with Lanczos, so opposite
edges match at the wrap. Planar mapping uses MuJoCo `texuniform=true` with a metric `texrepeat`. The surfaces stay
ordinary collision primitives: dimensions, friction, mass and contacts are unchanged.
