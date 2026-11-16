# G1 robot assets

The one robot model used by forward kinematics, the collision checks, the simulated camera and the MuJoCo bridge.

- `g1.xml`: the 29-DoF G1; masses, inertias, joints and actuators are those of the robot's own model. The torso's
  collision geometry is the visual mesh `torso_link` (MuJoCo uses its convex hull); the other links use the collision
  shapes in the XML.
- `meshes/`: the meshes of `g1.xml` and of the hand models, referred to by relative path.
- `LICENSE`: Unitree's BSD 3-Clause licence. `g1.xml`, `meshes/` and `hands/dex3_1.xml` come from Unitree's G1
  description files (unitreerobotics/unitree_ros). Our changes to the model (collision shapes, the separate hand
  models) are released under the same licence; the rest of the repository is MIT (`LICENSE` at the root).

The model path is `wbc.paths.G1_ASSETS`; `wbc.robot.model` assembles the scene objects and the collision setup.
