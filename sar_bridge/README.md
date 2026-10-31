# PyBullet backend + Unity frontend bridge

```
 PyBullet (headless, Python)  ──TCP 127.0.0.1:5555, JSON lines──►  Unity (render + operator input)
   physics, sensing, mission  ◄────────────  commands  ──────────  no simulation logic
```

PyBullet is the single source of truth. Unity never moves the drone on its own.

## Files

| File | Side | Purpose |
|---|---|---|
| `bridge.py` | Python | TCP server, replays hello/path/prob/detection to late joiners |
| `terrain.py` | Python | Terrain contract (raster + json). Swap `make_synthetic()` for the real DTM export |
| `sar_demo.py` | Python | Headless sim + lawnmower flight + prob grid + detection. `HOOK` comments mark where your Navigator / SearchMonitor / ProbabilityMap / Mission plug in |
| `test_client.py` | Python | Pretends to be Unity for protocol testing |
| `unity/CoordConv.cs` | Unity | The only place PyBullet <-> Unity axes are converted |
| `unity/BridgeClient.cs` | Unity | Background-thread TCP client with auto-reconnect |
| `unity/SarSceneController.cs` | Unity | Builds terrain, moves drone, draws path + heatmap + detection, sends commands |

## Run it

1. `pip install pybullet numpy`, then `python sar_demo.py` (or `--no-pybullet` to test only the bridge).
   First run writes `terrain_out/terrain.raw` and `terrain.json`.
2. Copy both files from `terrain_out/` into `Assets/StreamingAssets/` in Unity. Re-copy whenever the terrain changes.
3. Unity scene (3D project): put the 4 `.cs` files in `Assets/Scripts/`. Then
   - empty GameObject "Bridge" with `BridgeClient` + `SarSceneController`
   - a Sphere named "Drone" (remove its collider) -> assign to `drone`
   - a LineRenderer (width ~1.5) -> `pathLine`
   - a Quad with a transparent material (URP/Unlit or Sprites/Default) -> `heatmapQuad`
   - a small red Sphere/Cylinder, inactive -> `detectionMarker`
4. Press Play. Click = send drone there, Shift+click = set last-known position, P / R = pause / resume.
   Uses the legacy `Input` class; if the project defaults to the new Input System, set
   Player Settings -> Active Input Handling to "Both".

## Message schema

Backend -> Unity (all positions in PyBullet coordinates, meters):

- `hello`: `{terrain:{n,size_m,cell_m,h_min,h_max,origin_lat,origin_lon}, state_hz, grid}` sent once on connect
- `state`: `{t, paused, pos:[x,y,z], quat:[x,y,z,w], remaining}` 20 Hz
- `path`: `{pts:[x,y,z, x,y,z, ...]}` flat, whenever the plan changes
- `prob`: `{n, data:[n*n floats]}` row = north, col = east, about 2 Hz
- `detection`: `{pos:[x,y,z], conf}`

Unity -> backend: `{"type":"cmd","cmd":"goto"|"set_lkp"|"pause"|"resume","pos":[x,y]}`

Arrays are flat because Unity's `JsonUtility` can't parse nested arrays.

## Verified vs. not verified

Verified here: the bridge, protocol, late-join replay, and command round-trip (`test_client.py` against `--no-pybullet`).

NOT verified (no PyBullet or Unity in the sandbox where this was written), so check these first:

1. **PyBullet heightfield alignment.** `PyBulletWorld` shifts the heightfield body by half the terrain size and by the min/max midpoint height, and assumes `heightfieldData` is row-major with row = y. Test: `ground_height()` from `rayTest` should match `terrain.height_at()` at a few points. If it doesn't, the axes or offset are wrong. Fix it in that one class.
2. **Unity terrain orientation.** Put a marker at a known PyBullet coordinate (e.g. the person at `(0.7*size, 0.55*size)`) and confirm it sits on the same hill in Unity. If the terrain looks mirrored, transpose the loop in `BuildTerrain`.
3. **Heatmap overlay orientation** and the Unity C# in general: compiled by eye, not by Unity.

## Next steps

- Replace `make_synthetic()` with the DTM -> raster export (the geo-data task), keeping the `terrain.raw` / `terrain.json` contract.
- Move the HOOK sections onto your real classes; the drone body in PyBullet is a placeholder sphere.
- A real heatmap would be a decal/projector draped on the terrain instead of a floating quad.
