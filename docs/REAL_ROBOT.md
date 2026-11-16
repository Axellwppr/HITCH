# Real robot

**The controller adds no safety limits of its own. An operator must supervise the robot and hold the remote's
emergency stop. The harness has no stop command: motion already queued plays to its end.**

- [Hardware](#hardware)
- [Build](#build)
- [Start the robot side](#start-the-robot-side)
- [Depth repair](#depth-repair)
- [Run a model](#run-a-model)

## Hardware

- A Unitree G1 with rubber hands (`WBC_HAND=rubber`, the default).
- A RealSense D455 on a two-axis servo head ([vision/README.md](../vision/README.md); CAD, 3D-printed parts and
  wiring in the
  [head assembly files](https://drive.google.com/drive/folders/1Sjem4y6XnZZKC2NSFvBJp-6FXwfXIiPq?usp=drive_link)).
  Connect the camera over USB 3.0+ to the G1's top-left USB-C port (on USB 2.0 it fails with `Couldn't resolve
  requests`) and the servo adapter to any USB port.
- The robot computer: the computer on the G1's DDS network, the G1's own or an external PC. How to connect it is in
  the controller's [G1 Network Setup](https://github.com/Axellwppr/motion_tracking/tree/sim-to-real-harness#g1-network-setup).

## Build

Once, on the robot computer (librealsense must be installed):

```bash
scripts/setup
scripts/robot build
sudo cp vision/99-wbc-vision.rules /etc/udev/rules.d/    # the camera's and the servo adapter's device rules
sudo udevadm control --reload-rules && sudo udevadm trigger --action=add
```

## Start the robot side

One terminal per program on the robot computer, in this order:

```bash
sudo scripts/robot bridge IFACE  # IFACE: the network interface on the robot's DDS network (e.g. eth0)
scripts/robot controller         # then press start, and A, on the remote to enter tracking
scripts/robot link               # link server, port 18767; once the controller is tracking
scripts/robot vision             # camera server, port 18764
```

The remote's buttons: start moves the robot from zero torque to the default pose, A enters tracking, select stops the
controller ([Control Buttons](https://github.com/Axellwppr/motion_tracking/tree/sim-to-real-harness#control-buttons)
in the controller's README).

Pressing space in the terminal of `scripts/robot link` pauses for good: motion already handed to the controller plays
to its end, and later motion is not passed on. Restart the link server to continue.

## Depth repair

The D455's depth has missing regions. [LingBot-Depth](../scripts/lingbot/README.md) repairs it on a GPU machine:

```bash
bash scripts/lingbot/setup.sh    # once: installs ~/lingbot
~/lingbot/serve.sh 0 18890       # GPU 0, port 18890, on 127.0.0.1
```

Set `WBC_DEPTH_REFINE=http://127.0.0.1:18890` where the harness runs; if the service is on another machine, forward
the port first (`ssh -L 18890:127.0.0.1:18890 GPU_HOST`). If a request to it fails, the observation reports the
camera as failed and the agent stops.

## Run a model

The link and camera servers listen on 127.0.0.1. From another machine, forward both ports
(`ssh -L 18767:127.0.0.1:18767 -L 18764:127.0.0.1:18764 ROBOT`). `--host` binds another address, but the servers
have no authentication: anyone who reaches them can move the robot.

The built-in task takes its sentence from the simulated scene, so write a task file:

```markdown
---
target: http://127.0.0.1:18767
max_turns: 40
---
Grab the brown box on the rectangular table in front of you with both hands and hold it up.
```

```bash
export WBC_DEPTH_REFINE=http://127.0.0.1:18890
scripts/agent run my_task.md --backend hitch --vision http://127.0.0.1:18764
```

To send the steps yourself instead, `scripts/wbc connect http://127.0.0.1:18767 --vision http://127.0.0.1:18764`
opens a session ([USAGE.md](USAGE.md#session-scriptswbc)).
