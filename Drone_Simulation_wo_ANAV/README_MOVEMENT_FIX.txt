Drone simulation movement fix

Run from this folder:
  python main.py

Press START MISSION in the PyBullet debug panel.
Headless test:
  python main.py --headless --max-time 240

Changes in main.py:
- explicit fixed timestep and disabled real-time simulation
- smoother chase-camera updates (every 4 physics steps instead of every 30)
- wider camera view to keep the drone visible
- removed repeated velocity resets for the kinematic drone
- preserved speed across destination changes to avoid sudden acceleration resets

Requires: pip install pybullet
This remains a kinematic simulation, not motor-level quadcopter physics.
