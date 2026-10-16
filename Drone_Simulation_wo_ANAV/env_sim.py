"""Shared PyBullet environment. Run main.py for the integrated mission."""
import os
import pybullet as p
import pybullet_data
from ObjectAvoidance import DRONE_START_POS, DRONE_SCALE, OBSTACLE_POSITIONS, FLIGHT_PATH


def build_world(gui=True, assets=None):
    """Create exactly one world, drone and set of trees; target spawned separately."""
    assets = assets or os.path.dirname(os.path.abspath(__file__))
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    floor_id = p.loadURDF("plane.urdf")
    texture = os.path.join(assets, "ground_texture.jpg")
    if gui and os.path.exists(texture):
        try:
            p.changeVisualShape(floor_id, -1, textureUniqueId=p.loadTexture(texture))
        except Exception as exc:
            print("Texture not applied:", exc)
    drone_id = p.loadURDF(os.path.join(assets, "custom_drone.urdf"), DRONE_START_POS,
                          p.getQuaternionFromEuler([0, 0, 0]), globalScaling=DRONE_SCALE)
    # Navigator moves the drone kinematically. Prevent gravity/contacts from
    # moving it a second time between position updates.
    p.changeDynamics(drone_id, -1, mass=0)
    obstacles = []
    for x, y, z in OBSTACLE_POSITIONS:
        tc = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=2.5)
        tv = p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=2.5, rgbaColor=[0.4, 0.2, 0.05, 1])
        obstacles.append(p.createMultiBody(0, tc, tv, [x, y, z]))
        lc = p.createCollisionShape(p.GEOM_SPHERE, radius=0.75)
        lv = p.createVisualShape(p.GEOM_SPHERE, radius=0.75, rgbaColor=[0.1, 0.45, 0.1, 1])
        obstacles.append(p.createMultiBody(0, lc, lv, [x, y, z + 1.75]))
    return floor_id, drone_id, obstacles


if __name__ == "__main__":
    print("Run main.py to launch the integrated simulation.")
