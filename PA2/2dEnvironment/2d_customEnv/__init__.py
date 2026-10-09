from gymnasium.envs.registration import register

register(
    id="2d_customEnv/GridWorld-v0",
    entry_point="2d_customEnv.envs:GridWorldEnv",
)
