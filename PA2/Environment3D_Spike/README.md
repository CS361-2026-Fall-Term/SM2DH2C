# 3D Environment Simulation Spike

Technical Question: Can Pybullet simulate a static 3D environment with a ground
with a ground surface that contains obstacles, the target, and the drone? Is it
able to define a start position for each of the listed objects?

Technical Risk/Uncertainty: Simplifies real-world physics mechanisms, so complex
physical factod like friction and aerodynamics may be harder to simulate.

Planned Spike: Interface test, testing the physics engine framework.

Status: Completed.

Initial Decision: Use and extend PyBullet. Having the environment represented in
3D allows us to better design natural terrrains, in addition to simulating
better physics components.
