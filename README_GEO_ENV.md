# SM2DH2C — Geographic PyBullet Environment

The replacement `env_sim.py` builds either the original handwritten environment **or** a geographic environment generated from OpenStreetMap data, while returning the original `(floor_id, drone_id, obstacle_ids)` API.

## NEW: GeoPandas parser + JSON export (`osm_parser.py`)

All geographic processing now runs through pandas/GeoPandas GeoDataFrames in `osm_parser.py`
(OSMnx is only the download client and hands back a GeoDataFrame). Pipeline:

1. `prompt_for_lkp()` asks for latitude, longitude and radius (validated, re-prompts on bad input)
2. `fetch_osm()` downloads features (or `read_geojson()` for offline files)
3. `to_local_frame()` reprojects to meters centered on the LKP (0,0) and clips to a true circle
4. `classify()` labels each row `building / road / tree / water / forest / grass`, parses heights
5. `export_json()` writes `environment.json` (metadata + features, coordinates in local meters)

```bash
python main.py --prompt                      # asks for lat/lon/radius, builds from OSM, saves environment.json
python main.py --json-in environment.json    # rebuild the same world offline
python env_sim.py --prompt --no-gui          # generator only
python osm_parser.py                         # parse + export JSON without PyBullet
```

JSON shape: `{"metadata": {lkp, radius_m, local_crs, units, feature_counts, attribution},
"features": [{"id", "category", "tags", "height_m"?, "geometry": GeoJSON-style}]}`.

## Install and run

Place `env_sim.py` alongside your existing `custom_drone.urdf`. Use a Python virtual environment and install:

```bash
python -m pip install pybullet geopandas osmnx shapely pyproj
```

Online, with geographic coordinates from the missing person's last known position (LKP):

```bash
python env_sim.py --lat 44.5646 --lon -123.2620 --radius 30
```

For a fast build in headless mode:

```bash
python env_sim.py --lat 44.5646 --lon -123.2620 --radius 30 --no-gui
```

For prompted location input:

```bash
python env_sim.py --prompt
```

For a completely offline sample (synthetic geometry, NOT real OSM map data):

```bash
python env_sim.py --lat 44.5646 --lon -123.2620 --radius 30 --geojson example_synthetic_region.geojson
```

To view your original hardcoded scene:

```bash
python env_sim.py --demo
```

## Using your existing integrated `main.py`

Your existing `main.py` calls `build_world(gui)` without geographic inputs. The new file reads these optional environment variables, so **no direct change to `main.py` is required to make it build a geographic scene**:

macOS / Linux:

```bash
SAR_LAT=44.5646 SAR_LON=-123.2620 SAR_RADIUS_M=10 python main.py
```

Windows PowerShell:

```powershell
$env:SAR_LAT = '44.5646'
$env:SAR_LON = '-123.2620'
$env:SAR_RADIUS_M = '10'
python main.py
```

Or from Python:

```python
from env_sim import build_world
floor_id, drone_id, obstacle_ids = build_world(
    gui=True, lat=44.5646, lon=-123.2620, radius_m=30
)
```

**Important limitation:** This `env_sim.py` implements the **environment generator**, not changes to the rest of the mission system. `object_avoidance.py` has a fixed +/-13-meter A* grid, and `operator_guidance.py` still defines zones at (+/-6, +/-6). To navigate correctly over geographic scenes bigger than roughly 10m radius, change those modules and the probability-map rendering in `main.py`. The existing mission remains a small-area demonstration; a large geographical scene can currently be inspected directly with `env_sim.py`.

## What the geographic mode creates

- Map features around an LKP downloaded via OSMnx (uses cached Overpass responses where available)
- Locally projected coordinates in meters, with the LKP at `(0, 0)`
- Buildings extruded from OSM footprints into 3D static collision meshes
- Roads and paths visualized as flat segments; **visual only** (not obstacles)
- Mapped individual trees (collision bodies)
- Forest *areas* filled with deterministic **approximate**, not verified, tree positions
- Water, meadow and park land-use surfaces (**visual only**)
- A safe drone starting position and missing-person target point, placed outside mapped buildings and tree crowns
- Original scene when no geographical input is given

**Limitations:** OpenStreetMap does not provide a full digital elevation model, so this version uses a flat ground. Default building heights are estimates when height tags are absent. OSM vegetation/building coverage and geometry vary by location. OSMnx needs a working connection to OpenStreetMap/Overpass to download new areas. The generated forest tree positions are synthetic approximations, not mapped real-world tree coordinates. Some concave footprints may triangulate less precisely with Shapely older than 2.1.

A private/local GeoJSON FeatureCollection is supported via `--geojson`. This expects WGS84 coordinates in longitude/latitude order (as standard GeoJSON). **Raw `.osm` XML is not the same format and is not supported by the `--geojson` option.**

OSM map attribution: © OpenStreetMap contributors, distributed under the ODbL — https://www.openstreetmap.org/copyright

## Validation

- Python syntax compilation: passed.
- Offline geometry smoke test using a stand-in PyBullet API: passed (synthetic GeoJSON generated buildings, road segments, land-use surfaces, mapped and approximate trees; legacy API preserved).
- **Not tested**: live OSMnx network call or actual native PyBullet collision/raycast/GUI rendering, because PyBullet and OSMnx were not installed in the validation runtime and network installation was blocked. Test these in your development environment before merging.

References:
- OSMnx: https://osmnx.readthedocs.io/en/stable/user-reference.html
- PyBullet quickstart: https://github.com/bulletphysics/bullet3/blob/master/docs/pybullet_quickstart_guide/PyBulletQuickstartGuide.md.html
- PyBullet mesh example: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/createMesh.py
