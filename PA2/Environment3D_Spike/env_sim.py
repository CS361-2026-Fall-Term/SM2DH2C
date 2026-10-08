import pybullet as p
import pybullet_data
import time
import math

# Physics engine is started w/ 3D GUI window
physicsCilent = p.connect(p.GUI)

# Add standard folders that gives access to basic shapes & textures
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity to standard Earth gravity
p.setGravity(0, 0, -9.81)

# Load ground plane
floor_id = p.loadURDF("plane.urdf")

# Load texture image file, give error message and default to gray if loading fails
try: 
    texture_id = p.loadTexture("ground_texture.jpg")
    p.changeVisualShape(floor_id, -1, textureUniqueId=texture_id)
    print("Success: Applied ground texture.")
except Exception as e:
    print("Fail: Could not load image file. Defaulting to gray grid.")
    print("Error details:", e)

# Intialize drone and load it in the simulation (using custom drone .urdf)
drone_start_pos = [-10, 10, 3.5] # start position of drone (x, y, z)
drone_start_orientation = p.getQuaternionFromEuler([0, 0, 0]) # set inital drone orientation 
drone_id = p.loadURDF("custom_drone.urdf", drone_start_pos, drone_start_orientation, globalScaling=1.5) # load drone
 
# Set obstacle positions
obstacle_positions = [
    [1.5, 6.5, 0.5],
    [-4.0, -1.5, 0.5],
    [8.0, -2.0, 0.5],
    [-7.5, 5, 0.5],
    [-8.0, -8.0, 0.5],
    [9.0, 8.5, 0.5],
    [-1.0, -7.5, 0.5],
    [0.0, 1.0, 0.5]
]

# Create and load tree obstacles in the simulation
# Type of obstacles are subject to change (e.g., infrastructure, unsafe zones, etc.)
for x, y, z in obstacle_positions:
    # Trunk
    trunk_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=0.2,
        height=2.5
    )
    trunk_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.2,
        length=2.5,
        rgbaColor=[0.4, 0.2, 0.05, 1]
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=trunk_collision,
        baseVisualShapeIndex=trunk_visual,
        basePosition=[x, y, z]
    )

    # Leaves
    leaves_collision = p.createCollisionShape(
        p.GEOM_SPHERE,
        radius=0.75
    )
    leaves_visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.75,
        rgbaColor=[0.1, 0.45, 0.1, 1]
    )
    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=leaves_collision,
        baseVisualShapeIndex=leaves_visual,
        basePosition=[x, y, z + 1.75]
    )


# r2d2 is a temporary robot that acts as an obstacle, might be used for detection testing
# for pos in obstacle_positions:
#     p.loadURDF("r2d2.urdf", pos) 


# Set humanoid target position and load target (using built-in humaniod .urdf)
target_position = [7.0, -5.5, 0.875] # set target position
target_orientation = p.getQuaternionFromEuler([1.5708, 0, 0]) # set intial target orientation
target_id = p.loadURDF("humanoid/humanoid.urdf", target_position, target_orientation, globalScaling=0.25, useFixedBase=True) # load target

# Message when simulation is initialized
print(f"Simulation is initialized. Done ID: {drone_id} Target ID: {target_id}")

# Set flight path of the drone 
# Note: This is just for simulation purposes, will make the drone movement autonomous in the system prototype
flight_path = [
    [-10, 10, 2.5],
    [-5, 10, 2.5],
    [-5, 5, 2.5],
    [-5, 0, 2.5],
    [0, 0, 2.5],
    [5, 0, 2.5],
    [5, 5, 2.5],
    [10, 5, 2.5],
    [10, -5, 2.5],
    [5, -5, 2.5],
    [6, -5, 2.5]
]

# Initialize variables used in drone flight simulation
segment = 0
progress = 0.0
flight_speed = 0.0025

# Run continuous simulation loop
try:
    while True:
        # Drone flight simulation
        if segment < len(flight_path) - 1:
            start = flight_path[segment]
            end = flight_path[segment + 1]
            progress += flight_speed

            if progress >= 1.0:
                progress = 0.0
                segment += 1
            else:
                position = [
                    start[i] + progress * (end[i] - start[i])
                    for i in range(3)
                ]
                p.resetBasePositionAndOrientation(drone_id, position, drone_start_orientation)

        # Follow drone with camera
        drone_position, _ = p.getBasePositionAndOrientation(drone_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=drone_position
        )

        # Step physics engine forward one frame
        p.stepSimulation()

        #TODO can add autonous drone logic/keyboard controls here

        # Keep simulation running smoothly at around 60Hz
        time.sleep(1./240.)

except KeyboardInterrupt:
    # Exit when script is stopped
    p.disconnect()