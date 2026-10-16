"""Optional standalone DSSE experiment; not part of the PyBullet physics loop."""
from DSSE import DroneSwarmSearch

def main():
    env = DroneSwarmSearch(
        grid_size=40, render_mode="human", render_grid=True, render_gradient=True,
        vector=(1, 1), timestep_limit=300, person_amount=4,
        dispersion_inc=0.05, person_initial_position=(15, 15),
        drone_amount=2, drone_speed=10, probability_of_detection=0.9,
        pre_render_time=0,
    )
    options = {
        "drones_positions": [(10, 5), (10, 10)],
        "person_pod_multipliers": [0.1, 0.4, 0.5, 1.2],
        "vector": (0.3, 0.3),
    }
    observations, info = env.reset(options=options)
    done = False
    while not done:
        actions = {agent: env.action_space(agent).sample() for agent in env.get_agents()}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        done = any(terminations.values()) or any(truncations.values())
    env.close()

if __name__ == "__main__":
    main()
