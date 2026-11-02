"""
File Description: Single source of truth for the size of the simulated search map.

The map is a square [-MAP_HALF, MAP_HALF] in x and y (sim metres). The size is set ONCE at
startup by main.py (set_map_size) BEFORE the world, navigator, or probability map are built.

Other modules must read it as `sim_config.MAP_HALF` (attribute lookup at call time), NOT
`from sim_config import MAP_HALF` (that copies the number at import time and never updates).
"""

BASE_MAP_SIZE = 26.0           # Size the hand-placed layout (trees, zones, start) was designed for.
MIN_MAP_SIZE = 20.0            # Below this, zones / start / keep-clear areas start to overlap.
MAX_MAP_SIZE = 60.0            # Above this, A* grid and heatmap cost grow too much for a smooth GUI.

# Real-world width of the search area (--area). Above MAX_MAP_SIZE the OSM data is squeezed into
# the map; at 500 m on a 60 m map that is scale 0.12 (buildings become small, heights are clamped).
MIN_AREA_M = MIN_MAP_SIZE
MAX_AREA_M = 500.0

MAP_HALF = BASE_MAP_SIZE / 2


def set_map_size(size):
    """Set the map width in sim metres. Raises ValueError outside [MIN_MAP_SIZE, MAX_MAP_SIZE]."""
    global MAP_HALF
    if not MIN_MAP_SIZE <= size <= MAX_MAP_SIZE:
        raise ValueError(f"map size must be between {MIN_MAP_SIZE:g} and {MAX_MAP_SIZE:g} m (got {size:g})")
    MAP_HALF = size / 2.0
    return MAP_HALF


def map_size():
    return 2.0 * MAP_HALF


def layout_scale():
    """How much bigger the map is than the layout's original 26 m (1.0 = unchanged)."""
    return map_size() / BASE_MAP_SIZE


def clamp_map_size(size):
    return max(MIN_MAP_SIZE, min(MAX_MAP_SIZE, size))
