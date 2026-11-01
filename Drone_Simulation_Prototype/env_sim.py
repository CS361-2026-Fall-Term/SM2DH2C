"""
Contributors: Mari Tokuda and Heeseo Kim
File Description: Creates shared PyBullet environment. Builds the physical world where
the search happens, including the visualization of the drone body, trees, and ground.
While the target position is set here, the humaniod target is created and spawned 
separately in target_detection.py.
"""
import os
import pybullet as p
import pybullet_data

# Static positions of drone, obstacle, and target in the simulated environment.
# Positions are (x , y, z) in the PyBullet world.
DRONE_START_POS = [-10, 10, 3.5]
DRONE_SCALE = 1.5
OBSTACLE_POSITIONS = [
    [1.5, 6.5, 0.5], [-4.0, -1.5, 0.5], [8.0, -2.0, 0.5], [-7.5, 5, 0.5],
    [-8.0, -8.0, 0.5], [9.0, 8.5, 0.5], [-1.0, -7.5, 0.5], [0.0, 1.0, 0.5],
]
TARGET_POSITION = [7.0, -5.5, 0.875]
TARGET_ORIENTATION_EULER = [1.5708, 0, 0]
TARGET_SCALE = 0.25
FLIGHT_PATH = [
    [-10, 10, 2.5], [-5, 10, 2.5], [-5, 5, 2.5], [-5, 0, 2.5], [0, 0, 2.5],
    [5, 0, 2.5], [5, 5, 2.5], [10, 5, 2.5], [10, -5, 2.5], [5, -5, 2.5],
    [6, -5, 2.5],
]


def build_world(gui=True, assets=None):
    """
    Create exactly one world, drone and set of trees; target spawned separately.
    gui: True opens PyBullet window, False runs with no window (headless).
    assets: Folder holding custom_drone.urdf and ground_texture.jpg (defaults to the folder this file is in).
    Returns (floor_id, drone_id, obstacle_ids): PyBullet body ids for the ground, the drone and every tree part.
    """
    assets = assets or os.path.dirname(os.path.abspath(__file__))
    # Start PyBullet: GUI = window to watch simulation, DIRECT = no window
    p.connect(p.GUI if gui else p.DIRECT) 
    # PyBullet searchs built-in models, including plane.urdf used for the floor_id.
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81) 

    # Initialize floor (ground) of simulated environment.
    floor_id = p.loadURDF("plane.urdf")
    texture = os.path.join(assets, "ground_texture.jpg")
    if gui and os.path.exists(texture):
        try:
            p.changeVisualShape(floor_id, -1, textureUniqueId=p.loadTexture(texture))
        except Exception as exc:
            print("Texture not applied:", exc)

    # Intialize drone body, loaded from custom URDF file. 
    drone_id = p.loadURDF(os.path.join(assets, "custom_drone.urdf"), DRONE_START_POS,
                          p.getQuaternionFromEuler([0, 0, 0]), globalScaling=DRONE_SCALE)
    # Navigator moves the drone kinematically. Prevent gravity/contacts from moving it a second time between position updates.
    p.changeDynamics(drone_id, -1, mass=0)

    # Intialize obstacle trees. Each tree has 2 separate fixed bodies (trunk and canopy).
    # Tree is manually created, and have a collision shape so they are detectable. 
    obstacle_ids = []
    for x, y, z in OBSTACLE_POSITIONS:
        tc = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=2.5)
        tv = p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=2.5, rgbaColor=[0.4, 0.2, 0.05, 1])
        obstacle_ids.append(p.createMultiBody(0, tc, tv, [x, y, z]))
        lc = p.createCollisionShape(p.GEOM_SPHERE, radius=0.75)
        lv = p.createVisualShape(p.GEOM_SPHERE, radius=0.75, rgbaColor=[0.1, 0.45, 0.1, 1])
        obstacle_ids.append(p.createMultiBody(0, lc, lv, [x, y, z + 1.75]))

    return floor_id, drone_id, obstacle_ids


if __name__ == "__main__":
    print("Run main.py to launch the integrated simulation.")
