"""
File Description: Operator mission-setup window (issue #55).

Shown BEFORE the PyBullet simulation starts, so the operator enters the initial information
(last known position, size of the search area, missing person, zones to exclude, map data
source, display options) in a window
instead of on the command line. The window:
  - validates every input and shows a message next to each one (green = accepted, red = error),
  - loads the map data for the last known position (download or exported file) and reports
    what was found, or shows the error so the operator can fix it and try again,
  - keeps "Start simulation" disabled until all inputs have been validated,
  - returns the accepted settings to main.py, which builds the environment from them.

The window runs in its own process (main.py launches this file), so Tkinter and the PyBullet
window never share an application loop (they can conflict on macOS).

Run on its own to try the window:  python operator_setup.py
"""
import json
import os
import subprocess
import sys
import tempfile
import threading

import sim_config
from operator_guidance import ZONES

try:                                            # PBI-03: categories come from the probability map code
    from terrain_prior import PERSON_CATEGORIES, DEFAULT_CATEGORY
    CATEGORIES = list(PERSON_CATEGORIES)
except ImportError:                             # numpy missing: still show the window
    CATEGORIES, DEFAULT_CATEGORY = ["hiker", "runner", "child", "elderly/dementia"], "hiker"

HERE = os.path.dirname(os.path.abspath(__file__))
LAST_SETUP = os.path.join(HERE, "last_setup.json")     # pre-fills the window next time

# Known places the operator can pick instead of typing coordinates.
PRESETS = {
    "Custom (type coordinates)": None,
    "OSU - The Valley Library": (44.565296, -123.276016),
}

DEFAULTS = {
    "mode": "osm",              # "osm" = real map around the last known position, "practice" = built-in world
    "lat": "", "lon": "",
    "area": "60",
    "source": "download",       # "download" | "file"
    "osm_file": "",
    "map_size": "",             # blank = automatic
    "seed": "0",
    "visuals": "detailed",      # "detailed" | "simple"
    "camera": "locked",         # "locked" | "follow" | "free"  (see main.CAMERA_MODES)
    "max_time": "",             # blank = automatic
    "person_category": DEFAULT_CATEGORY,   # PBI-03: selects the distance-ring table
    "exclude_zones": "",        # PBI-03: zone letters to exclude before the mission, e.g. "AC"
}

MIN_TIME, MAX_TIME = 10.0, 3600.0


# ======================================================================= validation (no GUI)
def parse_zones(text):
    """'A, c' / 'AC' / ['A', 'C'] -> ['A', 'C'] (sorted, no repeats). Raises ValueError for an
    unknown zone letter."""
    letters = text if isinstance(text, (list, tuple)) else str(text or "").replace(",", " ").split()
    if not isinstance(text, (list, tuple)) and len(letters) == 1 and len(letters[0]) > 1:
        letters = list(letters[0])              # "AC" -> ["A", "C"]
    zones = sorted({str(z).strip().upper() for z in letters if str(z).strip()})
    unknown = [z for z in zones if z not in ZONES]
    if unknown:
        raise ValueError(f"unknown zone {', '.join(unknown)} (zones are {', '.join(ZONES)})")
    return zones


def _number(text, kind=float):
    text = str(text).strip()
    if text == "":
        return None
    return kind(text)


def validate_inputs(raw):
    """
    Check the operator's inputs. raw: dict of strings (see DEFAULTS).
    Returns (config, messages):
      config   = dict ready for main.py, or None if anything is invalid
      messages = {field: (ok, text)} - one message per checked field, shown next to it
    """
    msg = {}
    cfg = {"lat": None, "lon": None, "osm_file": None, "map_size": None}
    osm = raw.get("mode", "osm") == "osm"

    # --- last known position (only needed for the real-map environment)
    if osm:
        for key, name, lo, hi in (("lat", "Latitude", -90.0, 90.0), ("lon", "Longitude", -180.0, 180.0)):
            try:
                v = _number(raw.get(key, ""))
                if v is None:
                    msg[key] = (False, f"{name} is required for a real-map search.")
                elif not lo <= v <= hi:
                    msg[key] = (False, f"{name} must be between {lo:g} and {hi:g}.")
                else:
                    cfg[key] = v
                    msg[key] = (True, f"{name} {v:.6f} accepted.")
            except ValueError:
                msg[key] = (False, f"{name} must be a number in decimal degrees (e.g. 44.5652).")

    # --- search area width
    lo_a, hi_a = (sim_config.MIN_AREA_M, sim_config.MAX_AREA_M) if osm else \
                 (sim_config.MIN_MAP_SIZE, sim_config.MAX_MAP_SIZE)
    area = None
    try:
        area = _number(raw.get("area", ""))
        what = "Search area" if osm else "Practice map size"
        if area is None:
            msg["area"] = (False, f"{what} is required.")
        elif not lo_a <= area <= hi_a:
            msg["area"] = (False, f"{what} must be {lo_a:g}-{hi_a:g} m.")
        else:
            cfg["area"] = area
    except ValueError:
        msg["area"] = (False, "Area must be a number of metres.")

    # --- optional sim-map size override
    try:
        ms = _number(raw.get("map_size", ""))
        if ms is not None and not sim_config.MIN_MAP_SIZE <= ms <= sim_config.MAX_MAP_SIZE:
            msg["map_size"] = (False, f"Map size must be {sim_config.MIN_MAP_SIZE:g}-"
                                      f"{sim_config.MAX_MAP_SIZE:g} m, or blank for automatic.")
        else:
            cfg["map_size"] = ms
            if ms is not None:
                msg["map_size"] = (True, f"Simulation map fixed at {ms:g} m.")
    except ValueError:
        msg["map_size"] = (False, "Map size must be a number of metres, or blank for automatic.")

    if "area" in cfg:                       # describe the accepted area (auto map size if override invalid)
        size = cfg["map_size"] or sim_config.clamp_map_size(cfg["area"])
        if osm:
            scale = size / cfg["area"]
            how = "true scale" if abs(scale - 1) < 1e-6 else f"shown at scale {scale:.2f} on a {size:g} m map"
            msg["area"] = (True, f"{cfg['area']:g} m x {cfg['area']:g} m search area ({how}).")
        else:
            msg["area"] = (True, f"{size:g} m x {size:g} m practice map.")

    # --- map data source
    if osm and raw.get("source") == "file":
        path = str(raw.get("osm_file", "")).strip()
        if not path:
            msg["osm_file"] = (False, "Choose an exported .osm or .json file, or select Download.")
        elif not os.path.isfile(path):
            msg["osm_file"] = (False, "File not found.")
        elif not path.lower().endswith((".osm", ".json")):
            msg["osm_file"] = (False, "Map file must be an OpenStreetMap .osm or Overpass .json export.")
        else:
            cfg["osm_file"] = path
            msg["osm_file"] = (True, f"Map file {os.path.basename(path)} found.")

    # --- seed and time limit
    try:
        seed = _number(raw.get("seed", ""), int)
        if seed is not None and seed < 0:
            raise ValueError
        cfg["seed"] = seed or 0
    except ValueError:
        msg["seed"] = (False, "Seed must be a whole number of 0 or more.")
    try:
        mt = _number(raw.get("max_time", ""))
        if mt is not None and not MIN_TIME <= mt <= MAX_TIME:
            msg["max_time"] = (False, f"Time limit must be {MIN_TIME:g}-{MAX_TIME:g} s, or blank for automatic.")
        else:
            cfg["max_time"] = mt
            if mt is not None:
                msg["max_time"] = (True, f"Mission time limit {mt:g} s.")
    except ValueError:
        msg["max_time"] = (False, "Time limit must be a number of seconds, or blank for automatic.")

    # --- missing person (PBI-03)
    category = str(raw.get("person_category") or DEFAULT_CATEGORY).strip()
    if category in CATEGORIES:
        cfg["person_category"] = category
        msg["person_category"] = (True, f"Person category: {category}.")
    else:
        msg["person_category"] = (False, f"Choose a person category ({', '.join(CATEGORIES)}).")

    # --- zones excluded before the mission (PBI-03); same rule as "Do not search": the search
    # needs at least one zone left
    try:
        zones = parse_zones(raw.get("exclude_zones", ""))
        if len(zones) >= len(ZONES):
            msg["exclude_zones"] = (False, "At least one zone must stay in the search.")
        else:
            cfg["exclude_zones"] = zones
            if zones:
                msg["exclude_zones"] = (True, f"Zone{'s' if len(zones) > 1 else ''} "
                                              f"{', '.join(zones)} will not be searched.")
    except ValueError as exc:
        text = str(exc)
        msg["exclude_zones"] = (False, text[:1].upper() + text[1:] + ".")

    cfg["simple_visuals"] = raw.get("visuals") == "simple"
    cam = raw.get("camera") or "locked"
    if cam not in ("locked", "follow", "free"):
        msg["camera"] = (False, "Choose a camera mode.")
    cfg["camera"] = cam
    cfg["free_camera"] = cam == "free"            # kept for older code that reads this key
    cfg["mode"] = "osm" if osm else "practice"
    if not osm:
        cfg["lat"] = cfg["lon"] = cfg["osm_file"] = None

    if any(not ok for ok, _ in msg.values()):
        return None, msg
    return cfg, msg


def _points(el):
    if el.get("type") == "node":
        return [(el["lat"], el["lon"])]
    geoms = [el.get("geometry") or []] + [m.get("geometry") or [] for m in el.get("members", [])]
    return [(g["lat"], g["lon"]) for geom in geoms for g in geom if g]


def summarize_osm(data, lat=None, lon=None, area=None):
    """Count the main feature types in the OSM data (for the acknowledgement). With lat/lon/area,
    only features that reach into the search area are counted (an exported file can cover more)."""
    import math
    counts = {"buildings": 0, "roads and paths": 0, "trees": 0, "parks, lots and water": 0}
    half = None if area is None else area / 2
    m_lat, m_lon = 111_132.0, 111_320.0 * math.cos(math.radians(lat or 0))
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if half is not None and not any(abs((p[0] - lat) * m_lat) <= half and abs((p[1] - lon) * m_lon) <= half
                                        for p in _points(el)):
            continue
        if "building" in tags:
            counts["buildings"] += 1
        elif "highway" in tags:
            counts["roads and paths"] += 1
        elif tags.get("natural") == "tree":
            counts["trees"] += 1
        elif any(k in tags for k in ("leisure", "landuse", "amenity", "natural", "waterway")):
            counts["parks, lots and water"] += 1
    return counts


def load_map_data(cfg):
    """Download (or read) the map data so problems show up in the setup window, not later.
    The download is cached, so the simulation reuses it without downloading again."""
    import geo_obstacles
    import osm_world
    # A file that cannot be read raises GeoDataError with a message for the window; single map
    # objects that are malformed are dropped here and counted, like the simulation will do.
    problems = []
    if cfg.get("osm_file"):
        data = geo_obstacles.load_geo_file(cfg["osm_file"], problems)
    else:
        data = osm_world.fetch_overpass(cfg["lat"], cfg["lon"], cfg["area"])
    data = geo_obstacles.validate(data, problems)
    cfg["map_problems"] = len(problems)
    counts = summarize_osm(data, cfg["lat"], cfg["lon"], cfg["area"])
    if cfg.get("osm_file") and not any(counts.values()):
        total = sum(summarize_osm(data).values())
        raise ValueError(
            "The map file has no mapped features inside this search area"
            + (f" (it has {total} elsewhere)" if total else "")
            + ". Check that the last known position is inside the exported area.")
    return counts


def _short_error(text):
    """Keep the useful part of a download/read error for the window: one line per server,
    without the command-line advice meant for the terminal."""
    lines, seen = [], set()
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith("Try again") or "--osm-file" in line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines[:6])


def describe(cfg, counts=None):
    """One-paragraph acknowledgement of the accepted setup."""
    if cfg["mode"] == "practice":
        size = cfg["map_size"] or sim_config.clamp_map_size(cfg["area"])
        text = f"Practice world, {size:g} m x {size:g} m."
    else:
        text = (f"Last known position {cfg['lat']:.6f}, {cfg['lon']:.6f}; "
                f"{cfg['area']:g} m search area.")
        if counts is not None:
            found = ", ".join(f"{n} {k}" for k, n in counts.items() if n)
            text += f" Map data loaded: {found or 'no mapped features (open ground)'}."
        if cfg.get("map_problems"):
            text += (f" Warning: {cfg['map_problems']} map object(s) could not be read and will be "
                     "left out.")
    text += f" Missing person: {cfg.get('person_category', DEFAULT_CATEGORY)}."
    if cfg.get("exclude_zones"):
        text += f" Excluded zones: {', '.join(cfg['exclude_zones'])}."
    return text


# ======================================================================= the window
def run_dialog(defaults=None):
    """Show the setup window. Returns the accepted config dict, or None if cancelled."""
    import tkinter as tk
    from tkinter import ttk, filedialog, font as tkfont

    values = dict(DEFAULTS)
    values.update({k: str(v) for k, v in (defaults or {}).items() if k in DEFAULTS and v is not None})
    result = {"cfg": None}

    root = tk.Tk()
    root.title("Drone Search & Rescue - Mission Setup")
    root.resizable(False, False)
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    GREEN, RED, GREY = "#1e7d32", "#c62828", "#555555"
    base = tkfont.nametofont("TkDefaultFont")
    head_font = base.copy()
    head_font.configure(size=max(13, base.cget("size") + 4), weight="bold")
    bold_font = base.copy()
    bold_font.configure(weight="bold")
    root._fonts = (head_font, bold_font)          # keep references alive
    # tick / cross marks, if the system font has them (plain text otherwise)
    has_marks = all(base.measure(ch) > base.measure(" ") // 2 for ch in "✓✗")
    OK_MARK, BAD_MARK = ("✓ ", "✗ ") if has_marks else ("OK - ", "ERROR - ")
    style.configure("Head.TLabel", font=head_font)
    style.configure("Sub.TLabel", foreground=GREY)
    style.configure("Ok.TLabel", foreground=GREEN)
    style.configure("Err.TLabel", foreground=RED)
    style.configure("Start.TButton", font=bold_font)

    var = {k: tk.StringVar(value=v) for k, v in values.items()}
    preset = tk.StringVar(value=next(iter(PRESETS)))

    frm = ttk.Frame(root, padding=16)
    frm.grid(sticky="nsew")
    ttk.Label(frm, text="Mission setup", style="Head.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
    ttk.Label(frm, text="Enter the initial search information. The simulation starts once every "
                        "input is validated.", style="Sub.TLabel", wraplength=640).grid(
        row=1, column=0, columnspan=4, sticky="w", pady=(0, 10))

    message = {}      # field -> label showing that field's validation message
    widgets_osm = []  # widgets only used for the real-map environment

    def msg_label(row, field):
        lab = ttk.Label(frm, text="", style="Sub.TLabel", wraplength=300)
        lab.grid(row=row, column=3, sticky="w", padx=(10, 0))
        message[field] = lab

    def section(row, text):
        ttk.Separator(frm).grid(row=row, column=0, columnspan=4, sticky="ew", pady=(10, 4))
        ttk.Label(frm, text=text, font=bold_font).grid(
            row=row + 1, column=0, columnspan=4, sticky="w")

    r = 2
    section(r, "Search environment"); r += 2
    ttk.Label(frm, text="Environment").grid(row=r, column=0, sticky="w")
    env = ttk.Frame(frm)
    env.grid(row=r, column=1, columnspan=2, sticky="w")
    ttk.Radiobutton(env, text="Real map around last known position", value="osm",
                    variable=var["mode"]).pack(side="left")
    ttk.Radiobutton(env, text="Practice world (no map data)", value="practice",
                    variable=var["mode"]).pack(side="left", padx=(10, 0))
    r += 1

    ttk.Label(frm, text="Known place").grid(row=r, column=0, sticky="w")
    cb = ttk.Combobox(frm, textvariable=preset, values=list(PRESETS), state="readonly", width=30)
    cb.grid(row=r, column=1, columnspan=2, sticky="w", pady=2)
    widgets_osm.append(cb)

    def on_preset(_e=None):
        pos = PRESETS.get(preset.get())
        if pos:
            var["lat"].set(f"{pos[0]:.6f}")
            var["lon"].set(f"{pos[1]:.6f}")
    cb.bind("<<ComboboxSelected>>", on_preset)
    r += 1

    for key, label, hint in (("lat", "Latitude (last known)", "e.g. 44.565296"),
                             ("lon", "Longitude (last known)", "e.g. -123.276016")):
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w")
        e = ttk.Entry(frm, textvariable=var[key], width=18)
        e.grid(row=r, column=1, sticky="w", pady=2)
        ttk.Label(frm, text=hint, style="Sub.TLabel").grid(row=r, column=2, sticky="w")
        widgets_osm.append(e)
        msg_label(r, key)
        r += 1

    area_label = ttk.Label(frm, text="Search area width (m)")
    area_label.grid(row=r, column=0, sticky="w")
    area_box = ttk.Spinbox(frm, textvariable=var["area"], from_=sim_config.MIN_AREA_M,
                           to=sim_config.MAX_AREA_M, increment=10, width=8)
    area_box.grid(row=r, column=1, sticky="w", pady=2)
    area_hint = ttk.Label(frm, text="", style="Sub.TLabel")
    area_hint.grid(row=r, column=2, sticky="w")
    msg_label(r, "area")
    r += 1

    ttk.Label(frm, text="Map data").grid(row=r, column=0, sticky="w")
    src = ttk.Frame(frm)
    src.grid(row=r, column=1, columnspan=2, sticky="w")
    rb1 = ttk.Radiobutton(src, text="Download (OpenStreetMap)", value="download", variable=var["source"])
    rb2 = ttk.Radiobutton(src, text="Exported file", value="file", variable=var["source"])
    rb1.pack(side="left")
    rb2.pack(side="left", padx=(10, 0))
    widgets_osm += [rb1, rb2]
    r += 1

    ttk.Label(frm, text="Map file").grid(row=r, column=0, sticky="w")
    file_row = ttk.Frame(frm)
    file_row.grid(row=r, column=1, columnspan=2, sticky="w")
    fe = ttk.Entry(file_row, textvariable=var["osm_file"], width=28)
    fe.pack(side="left")

    def browse():
        path = filedialog.askopenfilename(
            title="Choose an OpenStreetMap export",
            filetypes=[("OpenStreetMap export", "*.osm *.json"), ("All files", "*.*")])
        if path:
            var["osm_file"].set(path)
            var["source"].set("file")
    fb = ttk.Button(file_row, text="Browse...", command=browse)
    fb.pack(side="left", padx=(6, 0))
    widgets_osm += [fe, fb]
    msg_label(r, "osm_file")
    r += 1

    section(r, "Missing person and zones"); r += 2
    ttk.Label(frm, text="Person category").grid(row=r, column=0, sticky="w")
    ttk.Combobox(frm, textvariable=var["person_category"], values=CATEGORIES, state="readonly",
                 width=18).grid(row=r, column=1, sticky="w", pady=2)
    ttk.Label(frm, text="sets how far they may have travelled", style="Sub.TLabel").grid(
        row=r, column=2, sticky="w")
    msg_label(r, "person_category")
    r += 1
    ttk.Label(frm, text="Do not search").grid(row=r, column=0, sticky="w")
    zone_row = ttk.Frame(frm)
    zone_row.grid(row=r, column=1, columnspan=2, sticky="w")
    try:
        already = set(parse_zones(var["exclude_zones"].get()))
    except ValueError:
        already = set()
    zone_vars = {z: tk.BooleanVar(value=z in already) for z in ZONES}

    def zones_changed(*_):
        var["exclude_zones"].set("".join(z for z, v in zone_vars.items() if v.get()))
    for z, v in zone_vars.items():
        ttk.Checkbutton(zone_row, text=f"Zone {z}", variable=v, command=zones_changed).pack(
            side="left", padx=(0, 8))
    msg_label(r, "exclude_zones")
    r += 1

    section(r, "Display and advanced options"); r += 2
    ttk.Label(frm, text="Visual quality").grid(row=r, column=0, sticky="w")
    vq = ttk.Frame(frm)
    vq.grid(row=r, column=1, columnspan=2, sticky="w")
    ttk.Radiobutton(vq, text="Detailed", value="detailed", variable=var["visuals"]).pack(side="left")
    ttk.Radiobutton(vq, text="Simple (faster)", value="simple", variable=var["visuals"]).pack(side="left", padx=(10, 0))
    r += 1
    ttk.Label(frm, text="Camera").grid(row=r, column=0, sticky="w")
    cm = ttk.Frame(frm)
    cm.grid(row=r, column=1, columnspan=2, sticky="w")
    for value, text in (("locked", "Follow drone (locked)"), ("follow", "Follow drone (orbit/zoom enabled)"),
                        ("free", "Free (mouse control)")):
        ttk.Radiobutton(cm, text=text, value=value, variable=var["camera"]).pack(
            side="left", padx=(0 if value == "locked" else 10, 0))
    r += 1
    for key, label, hint in (("map_size", "Sim map size (m)", "blank = automatic"),
                             ("max_time", "Time limit (s)", "blank = automatic"),
                             ("seed", "Random seed", "tree layout in woods")):
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=var[key], width=10).grid(row=r, column=1, sticky="w", pady=2)
        ttk.Label(frm, text=hint, style="Sub.TLabel").grid(row=r, column=2, sticky="w")
        msg_label(r, key)
        r += 1

    ttk.Separator(frm).grid(row=r, column=0, columnspan=4, sticky="ew", pady=(10, 6))
    r += 1
    status = ttk.Label(frm, text="Fill in the information, then press Validate inputs.",
                       style="Sub.TLabel", wraplength=640, justify="left")
    status.grid(row=r, column=0, columnspan=4, sticky="w")
    r += 1
    btns = ttk.Frame(frm)
    btns.grid(row=r, column=0, columnspan=4, sticky="e", pady=(10, 0))
    cancel_b = ttk.Button(btns, text="Cancel")
    validate_b = ttk.Button(btns, text="Validate inputs")
    start_b = ttk.Button(btns, text="Start simulation", style="Start.TButton", state="disabled")
    cancel_b.pack(side="left")
    validate_b.pack(side="left", padx=8)
    start_b.pack(side="left")

    state = {"cfg": None, "busy": False}

    def set_status(text, kind="sub"):
        status.configure(text=text, style={"ok": "Ok.TLabel", "err": "Err.TLabel"}.get(kind, "Sub.TLabel"))

    def show_messages(msgs):
        for field, lab in message.items():
            if field in msgs:
                ok, text = msgs[field]
                lab.configure(text=(OK_MARK if ok else BAD_MARK) + text, style="Ok.TLabel" if ok else "Err.TLabel")
            else:
                lab.configure(text="", style="Sub.TLabel")

    def refresh_mode(*_):
        osm = var["mode"].get() == "osm"
        for w in widgets_osm:
            w.configure(state=("readonly" if w is cb else "normal") if osm else "disabled")
        if osm and var["source"].get() != "file":
            fe.configure(state="disabled")
            fb.configure(state="normal")
        area_label.configure(text="Search area width (m)" if osm else "Practice map size (m)")
        lo, hi = ((sim_config.MIN_AREA_M, sim_config.MAX_AREA_M) if osm else
                  (sim_config.MIN_MAP_SIZE, sim_config.MAX_MAP_SIZE))
        area_box.configure(from_=lo, to=hi)
        area_hint.configure(text=f"{lo:g}-{hi:g} m")

    def inputs_changed(*_):
        if state["busy"]:
            return
        if state["cfg"] is not None:
            state["cfg"] = None
            start_b.configure(state="disabled")
            set_status("Inputs changed - press Validate inputs again before starting.")
        refresh_mode()

    for v in var.values():
        v.trace_add("write", inputs_changed)

    def set_busy(busy):
        state["busy"] = busy
        for b in (validate_b, cancel_b):
            b.configure(state="disabled" if busy else "normal")
        root.configure(cursor="watch" if busy else "")

    def validate():
        raw = {k: v.get() for k, v in var.items()}
        cfg, msgs = validate_inputs(raw)
        show_messages(msgs)
        if cfg is None:
            bad = sum(1 for ok, _ in msgs.values() if not ok)
            set_status(f"{BAD_MARK}{bad} input{'s' if bad > 1 else ''} need{'' if bad > 1 else 's'} fixing "
                       f"(see the red messages). Correct them and validate again.", "err")
            start_b.configure(state="disabled")
            return
        if cfg["mode"] == "practice":
            accept(cfg, None)
            return
        # real map: load the data now so a bad position or a network problem is caught here
        set_busy(True)
        set_status("Inputs valid. Loading map data for the last known position... "
                   "(the first download can take up to a minute)")
        box = {}

        def work():
            try:
                box["counts"] = load_map_data(cfg)
            except Exception as exc:                     # network down, servers busy, bad file
                box["error"] = str(exc)

        threading.Thread(target=work, daemon=True).start()

        def poll():
            if "counts" not in box and "error" not in box:
                root.after(150, poll)
                return
            set_busy(False)
            if "error" in box:
                field = "osm_file" if cfg.get("osm_file") else None
                if field:
                    message[field].configure(text=BAD_MARK + "Could not use this file.", style="Err.TLabel")
                    advice = "Fix the file or the position, then validate again."
                else:
                    advice = ("Check your internet connection and validate again in a minute, or export "
                              "the area from openstreetmap.org (Export button), choose 'Exported file' "
                              "and Browse to it.")
                set_status(BAD_MARK + "Map data could not be loaded.\n" + _short_error(box["error"])
                           + "\n" + advice, "err")
                start_b.configure(state="disabled")
                return
            accept(cfg, box["counts"])
        poll()

    def accept(cfg, counts):
        state["cfg"] = cfg
        cfg["summary"] = describe(cfg, counts)
        set_status(OK_MARK + "All inputs accepted. " + cfg["summary"] + "\nPress Start simulation to begin.", "ok")
        start_b.configure(state="normal")
        start_b.focus_set()

    def start():
        if state["cfg"] is None:                        # never start without validated inputs
            return
        result["cfg"] = state["cfg"]
        root.destroy()

    def cancel():
        result["cfg"] = None
        root.destroy()

    validate_b.configure(command=validate)
    start_b.configure(command=start)
    cancel_b.configure(command=cancel)
    root.protocol("WM_DELETE_WINDOW", cancel)
    root.bind("<Return>", lambda _e: start() if state["cfg"] else validate())
    refresh_mode()
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) // 3
    root.geometry(f"+{max(0, x)}+{max(0, y)}")
    root.mainloop()
    return result["cfg"]


# ======================================================================= used by main.py
def load_last():
    try:
        with open(LAST_SETUP) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_last(raw):
    try:
        with open(LAST_SETUP, "w") as f:
            json.dump(raw, f, indent=2)
    except OSError:
        pass


def cfg_to_raw(cfg):
    """Turn an accepted config back into window fields (to pre-fill the window next time)."""
    def blank(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6f}".rstrip("0").rstrip(".")      # keeps full coordinate precision
        return str(v)
    return {"mode": cfg["mode"], "lat": blank(cfg["lat"]), "lon": blank(cfg["lon"]),
            "area": blank(cfg["area"]), "source": "file" if cfg.get("osm_file") else "download",
            "osm_file": cfg.get("osm_file") or "", "map_size": blank(cfg["map_size"]),
            "seed": str(cfg["seed"]), "visuals": "simple" if cfg["simple_visuals"] else "detailed",
            "camera": cfg.get("camera") or ("free" if cfg.get("free_camera") else "locked"),
            "max_time": blank(cfg["max_time"]),
            "person_category": cfg.get("person_category", DEFAULT_CATEGORY),
            "exclude_zones": "".join(cfg.get("exclude_zones", []))}


class SetupUnavailable(RuntimeError):
    """The window could not be shown (no Tkinter / no display)."""


def ask(defaults=None):
    """
    Show the setup window in a separate process and wait for it.
    Returns the accepted config, None if the operator cancelled, or raises SetupUnavailable.
    """
    prefill = load_last()
    prefill.update({k: v for k, v in (defaults or {}).items() if v not in (None, "")})
    fd, out = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--out", out,
                               "--defaults", json.dumps(prefill)])
        if proc.returncode == 3:
            raise SetupUnavailable("the setup window needs Tkinter and a display")
        if proc.returncode != 0:
            return None
        with open(out) as f:
            cfg = json.load(f)
        save_last(cfg_to_raw(cfg))
        return cfg
    finally:
        try:
            os.remove(out)
        except OSError:
            pass


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Mission setup window")
    ap.add_argument("--out", help="write the accepted settings to this JSON file")
    ap.add_argument("--defaults", default="{}", help="JSON of initial field values")
    a = ap.parse_args()
    try:
        import tkinter  # noqa: F401
        cfg = run_dialog(json.loads(a.defaults))
    except Exception as exc:                      # ImportError (no Tkinter) or TclError (no display)
        if type(exc).__name__ in ("ImportError", "ModuleNotFoundError", "TclError"):
            print("Mission setup window unavailable:", exc, file=sys.stderr)
            sys.exit(3)
        raise
    if cfg is None:
        sys.exit(1)                               # cancelled
    if a.out:
        with open(a.out, "w") as f:
            json.dump(cfg, f)
    else:
        print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    _main()
