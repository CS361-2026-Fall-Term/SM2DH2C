"""Terrain contract shared by PyBullet and Unity.

One raster, one metadata file, used by BOTH sides so the drone never clips
through the ground:

  terrain.raw   float32 little-endian, N*N values, row-major.
                row index = north (y), column index = east (x), both increasing.
                values = elevation in meters.
  terrain.json  {"n": N, "size_m": S, "cell_m": S/(N-1), "h_min":..., "h_max":...,
                 "origin_lat":..., "origin_lon":...}
                World origin (0,0) is the SOUTH-WEST corner of the raster.

N should be 2^k + 1 (129, 257, 513) so Unity's TerrainData accepts it directly.

Replace make_synthetic() with the real DTM -> raster step from the
geo-data task; everything downstream stays the same.
"""
import json
import os

import numpy as np


def make_synthetic(n=257, size_m=512.0, seed=3):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xs, xs)  # X = east, Y = north
    h = np.zeros((n, n))
    for _ in range(6):
        fx, fy = rng.uniform(1, 4, 2)
        px, py = rng.uniform(0, 6.28, 2)
        h += rng.uniform(4, 14) * np.sin(fx * 6.28 * X + px) * np.cos(fy * 6.28 * Y + py)
    h -= h.min()
    return h.astype(np.float32), size_m


def save(path_dir, heights, size_m, origin_lat=44.5646, origin_lon=-123.2620):
    os.makedirs(path_dir, exist_ok=True)
    n = heights.shape[0]
    heights.astype("<f4").tofile(os.path.join(path_dir, "terrain.raw"))
    meta = {
        "n": int(n),
        "size_m": float(size_m),
        "cell_m": float(size_m) / (n - 1),
        "h_min": float(heights.min()),
        "h_max": float(heights.max()),
        "origin_lat": origin_lat,
        "origin_lon": origin_lon,
    }
    with open(os.path.join(path_dir, "terrain.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load(path_dir):
    with open(os.path.join(path_dir, "terrain.json")) as f:
        meta = json.load(f)
    n = meta["n"]
    h = np.fromfile(os.path.join(path_dir, "terrain.raw"), dtype="<f4").reshape(n, n)
    return h, meta


def height_at(heights, meta, x, y):
    """Bilinear ground height at world (x east, y north)."""
    n, cell = meta["n"], meta["cell_m"]
    fx = min(max(x / cell, 0), n - 1.001)
    fy = min(max(y / cell, 0), n - 1.001)
    i, j = int(fy), int(fx)
    ty, tx = fy - i, fx - j
    h00, h01 = heights[i, j], heights[i, j + 1]
    h10, h11 = heights[i + 1, j], heights[i + 1, j + 1]
    return float((h00 * (1 - tx) + h01 * tx) * (1 - ty) + (h10 * (1 - tx) + h11 * tx) * ty)
