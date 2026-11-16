# Tool descriptions

Each section is the description the model sees for the tool of that name. Input formats are also in each
parameter's schema description (tools.py); the rules are in system.md. Results are plain text written by render.py
and need no description here.

## execute

Plan a whole-body motion from where the robot is now, in frame O of the latest observation, and play it if it passes its checks.

First the in-place part: `height`, `torso` and `hands` start together with both feet planted (hands without waypoints `follow` the body as you choose). Then the fingers (`grip`, for hands with finger control), then the `walk` with the arms at their joint angles, then `wait_s`. Then the head camera turns to `eye`, for the new observation. Give at least one field; `eye` alone only turns the camera and observes again.

If the plan does not pass, nothing moves and your last observation still holds. If it passes, the motion is played and the result ends with the new observation, which defines a new frame O and says where it lies in the previous one.

## perceive

Measure selected image points in the latest observation. Give `points: [[x,y], ...]`, 1–100 points, using integer coordinates on both axes normalized to 0–1000. Origin is top-left, x right, y down; [500,500] is the image centre, [1000,1000] is bottom-right.

The result lists the points in your order with their 3D positions in O, and returns the image with the points marked and numbered from 1; check that each landed on the surface you meant.

## finish

End the episode; make it the only tool call of your last reply. `success`: true only if your latest observations show the task done. `summary`: what you did, what you saw at the end, and anything uncertain.
