"""OpenStreetMap -> GeoPandas -> JSON parser for the SAR drone simulation.

All geographic handling here is done with the pandas family only:
  * pandas / GeoPandas GeoDataFrames for tags, geometry, filtering, clipping,
    reprojection and export.
  * OSMnx is used ONLY as the download client; it returns a GeoDataFrame
    immediately and is not used for any geometry work.

Pipeline:
    prompt_for_lkp()  ->  fetch_osm()  ->  to_local_frame()  ->  classify()
                      ->  export_json()      (and later: load_json())

Local frame: azimuthal-equidistant projection centred on the last known
position (LKP), units = meters, LKP = (0, 0). This is what env_sim.py builds
its PyBullet world from.

OSM data (c) OpenStreetMap contributors, ODbL.
https://www.openstreetmap.org/copyright
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, mapping, shape

ATTRIBUTION = "(c) OpenStreetMap contributors, ODbL. https://www.openstreetmap.org/copyright"

OSM_TAGS = {
    "building": True,
    "highway": True,
    "natural": ["tree", "wood", "water", "scrub"],
    "landuse": ["forest", "grass", "meadow"],
    "leisure": ["park"],
    "waterway": ["riverbank"],
}
TAG_COLUMNS = ["building", "building:levels", "height", "highway", "natural",
               "landuse", "leisure", "waterway", "name"]

MIN_RADIUS_M, MAX_RADIUS_M = 5.0, 500.0


# --------------------------------------------------------------------------- #
# 1. User input
# --------------------------------------------------------------------------- #
def validate_lkp(lat: float, lon: float, radius_m: float) -> tuple[float, float, float]:
    if not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90.")
    if not -180 <= lon <= 180:
        raise ValueError("Longitude must be between -180 and 180.")
    if not MIN_RADIUS_M <= radius_m <= MAX_RADIUS_M:
        raise ValueError(f"Radius must be between {MIN_RADIUS_M:g} and {MAX_RADIUS_M:g} meters.")
    return lat, lon, radius_m


def prompt_for_lkp() -> tuple[float, float, float]:
    """Ask for the last known position and search radius; re-prompt on bad input."""
    while True:
        try:
            lat = float(input("Last known latitude  (e.g. 44.5646): "))
            lon = float(input("Last known longitude (e.g. -123.2620): "))
            raw = input(f"Search radius in meters [{MIN_RADIUS_M:g}-{MAX_RADIUS_M:g}, default 50]: ").strip()
            radius = float(raw) if raw else 50.0
            return validate_lkp(lat, lon, radius)
        except ValueError as exc:
            print(f"Invalid input: {exc}  Please try again.\n")


# --------------------------------------------------------------------------- #
# 2. Get raw features as a GeoDataFrame (WGS84)
# --------------------------------------------------------------------------- #
def fetch_osm(lat: float, lon: float, radius_m: float) -> gpd.GeoDataFrame:
    """Download OSM features around the LKP. Returns a WGS84 GeoDataFrame."""
    try:
        import osmnx as ox
    except ImportError as exc:
        raise RuntimeError("Install the OSM client: pip install osmnx") from exc
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(Path.home() / ".cache" / "sm2dh2c_osmnx")
    print(f"[osm_parser] Requesting OSM data: lat={lat:.6f} lon={lon:.6f} "
          f"radius={radius_m:g} m (needs internet/Overpass)...", flush=True)
    try:
        gdf = ox.features_from_point((lat, lon), tags=OSM_TAGS, dist=radius_m)
    except Exception as exc:  # network, rate limit, "no matching features", ...
        raise RuntimeError(
            "OpenStreetMap download failed (connection, Overpass limits, or no mapped "
            f"features here). Original error: {exc}") from exc
    print(f"[osm_parser] Downloaded {len(gdf)} raw features.", flush=True)
    return gdf


def read_geojson(path: str | Path) -> gpd.GeoDataFrame:
    """Offline alternative: a WGS84 GeoJSON FeatureCollection via GeoPandas."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(4326)


# --------------------------------------------------------------------------- #
# 3. Project to local meters, clip, classify
# --------------------------------------------------------------------------- #
def local_crs(lat: float, lon: float) -> str:
    return (f"+proj=aeqd +lat_0={lat:.10f} +lon_0={lon:.10f} "
            "+datum=WGS84 +units=m +no_defs")


def to_local_frame(gdf: gpd.GeoDataFrame, lat: float, lon: float,
                   radius_m: float) -> gpd.GeoDataFrame:
    """Reproject to LKP-centred meters and clip to a true circle."""
    gdf = gdf.reset_index()  # OSMnx puts element type + id in a MultiIndex
    gdf = gdf.rename(columns={"element_type": "element", "osmid": "id"})
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf.to_crs(local_crs(lat, lon))
    circle = gpd.GeoSeries([Point(0, 0).buffer(radius_m, 48)], crs=gdf.crs)
    gdf = gpd.clip(gdf, circle, keep_geom_type=False)
    gdf = gdf[~gdf.geometry.is_empty].copy()
    return gdf.explode(index_parts=False, ignore_index=True)  # one geometry per row


def _col(gdf: gpd.GeoDataFrame, name: str) -> pd.Series:
    """Tag column as a clean lowercase string Series ('' when missing/NaN)."""
    if name not in gdf.columns:
        return pd.Series("", index=gdf.index)
    return gdf[name].astype("string").fillna("").str.strip().str.lower().astype(str)


def _height_m(height: pd.Series, levels: pd.Series) -> pd.Series:
    """OSM 'height' (m or ft) -> meters; fall back to building:levels * 3 m; else 9 m."""
    num = height.str.extract(r"^\s*(\d+(?:\.\d+)?)")[0].astype(float)
    feet = height.str.contains(r"(?:ft|feet|')\s*$", regex=True)
    num = num.where(~feet, num * 0.3048)
    lv = pd.to_numeric(levels, errors="coerce")
    lv_h = (lv * 3.0).where((lv > 0) & (lv < 80))
    return num.fillna(lv_h).fillna(9.0).clip(2.0, 100.0)


def classify(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add 'category' + 'height_m'; keep only what the simulation uses."""
    building, highway = _col(gdf, "building"), _col(gdf, "highway")
    natural, landuse = _col(gdf, "natural"), _col(gdf, "landuse")
    leisure, waterway = _col(gdf, "leisure"), _col(gdf, "waterway")

    category = pd.Series("", index=gdf.index)
    # Later assignments lose to earlier ones: apply lowest priority first.
    category[landuse.isin(["grass", "meadow"]) | (leisure == "park")] = "grass"
    category[(natural == "wood") | (landuse == "forest")] = "forest"
    category[(natural == "water") | (waterway == "riverbank")] = "water"
    category[natural == "tree"] = "tree"
    category[highway != ""] = "road"
    category[building != ""] = "building"
    gdf = gdf.assign(category=category)
    gdf = gdf[gdf["category"] != ""].copy()

    # Geometry must match category (e.g. no polygon-shaped "road")
    gtype = gdf.geometry.geom_type
    ok = (
        ((gdf["category"] == "tree") & (gtype == "Point"))
        | ((gdf["category"] == "road") & (gtype == "LineString"))
        | (gdf["category"].isin(["building", "water", "forest", "grass"]) & (gtype == "Polygon"))
    )
    gdf = gdf[ok].copy()
    # Drop slivers created by clipping
    is_poly = gdf.geometry.geom_type == "Polygon"
    gdf = gdf[~(is_poly & (gdf.geometry.area < 2.0))]
    gdf = gdf[~((gdf.geometry.geom_type == "LineString") & (gdf.geometry.length < 0.1))]

    gdf["height_m"] = _height_m(_col(gdf, "height"), _col(gdf, "building:levels")).where(
        gdf["category"] == "building")
    keep = [c for c in ["element", "id", "category", "height_m", *TAG_COLUMNS, "geometry"]
            if c in gdf.columns]
    return gdf[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. JSON export / import
# --------------------------------------------------------------------------- #
def export_json(gdf: gpd.GeoDataFrame, lat: float, lon: float, radius_m: float,
                path: str | Path = "environment.json") -> Path:
    """Write a plain JSON file (coordinates in local meters, LKP at 0,0)."""
    features = []
    tag_cols = [c for c in TAG_COLUMNS if c in gdf.columns]
    for _, row in gdf.iterrows():
        tags = {c: str(row[c]) for c in tag_cols if pd.notna(row[c]) and str(row[c]) != ""}
        osm_id = f"{row['element']}/{row['id']}" if "id" in gdf.columns and pd.notna(row.get("id")) else None
        item = {
            "id": osm_id,
            "category": row["category"],
            "tags": tags,
            "geometry": mapping(row.geometry),
        }
        if pd.notna(row.get("height_m")):
            item["height_m"] = round(float(row["height_m"]), 2)
        features.append(item)

    doc = {
        "metadata": {
            "lkp": {"lat": lat, "lon": lon},
            "radius_m": radius_m,
            "local_crs": local_crs(lat, lon),
            "units": "meters",
            "origin": "last known position = (0, 0)",
            "feature_counts": gdf["category"].value_counts().to_dict(),
            "attribution": ATTRIBUTION,
        },
        "features": features,
    }
    path = Path(path)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def load_json(path: str | Path) -> tuple[gpd.GeoDataFrame, dict]:
    """Rebuild the local-meters GeoDataFrame from a file written by export_json()."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = doc["metadata"]
    rows = []
    for f in doc["features"]:
        rows.append({"id": f.get("id"), "category": f["category"],
                     "height_m": f.get("height_m"), **f.get("tags", {}),
                     "geometry": shape(f["geometry"])})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=meta["local_crs"]), meta


# --------------------------------------------------------------------------- #
# 5. One-call entry point used by env_sim.py
# --------------------------------------------------------------------------- #
def build_environment(lat: float, lon: float, radius_m: float, *,
                      geojson_path: str | None = None,
                      json_path: str | Path | None = "environment.json") -> gpd.GeoDataFrame:
    validate_lkp(lat, lon, radius_m)
    raw = read_geojson(geojson_path) if geojson_path else fetch_osm(lat, lon, radius_m)
    env = classify(to_local_frame(raw, lat, lon, radius_m))
    print("[osm_parser] Feature counts:", env["category"].value_counts().to_dict(), flush=True)
    if json_path:
        out = export_json(env, lat, lon, radius_m, json_path)
        print(f"[osm_parser] Wrote {out}", flush=True)
    return env


if __name__ == "__main__":
    la, lo, r = prompt_for_lkp()
    build_environment(la, lo, r)
