# Operator Guidance Spike (Diya, Helena)

Question: Can PyBullet's GUI show a live dashboard and accept preset
commands, and can the backend change the plan mid-mission?

Run: python3 -m pip install pybullet, then python3 operator_guidance_spike.py
Backend test: python3 operator_guidance_spike.py --selftest

Status: In progress. GUI results and decision to be added after testing.

Backend self-test: PASSED (run locally, 9 events logged).
Tested: preset commands change the plan, original plan can be resumed,
commands before START and after FOUND are rejected, every command
returns a response.
GUI dashboard/buttons: not yet tested (PyBullet install failing on
macOS, Python 3.13). To be tested on another team member's machine.
