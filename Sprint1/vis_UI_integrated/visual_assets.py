"""
File Description: Visual-only helpers that make the PyBullet scene look closer to the real place.
Nothing in this file creates collision shapes, so navigation, the LIDAR ray scan, and the
person-detection camera behave exactly the same with or without it.

  - write_png / facade_texture: procedurally generated wall textures (window grid per storey),
    written once to textures/ next to this file. No image library needed (numpy + zlib only).
  - MeshBuilder: collects triangles for many flat features (roads, grass, parking...) into a few
    large meshes, split into chunks so no single PyBullet mesh gets too big.
  - tree_canopy_mesh: low-poly deciduous (lumpy sphere) and conifer (stacked cones) canopies.
"""
import math
import os
import random
import struct
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEXTURE_DIR = os.path.join(HERE, "textures")

MAX_VERTS = 2000                  # per PyBullet mesh; bigger features are split across meshes
MAX_INDICES = 6000


# ------------------------------------------------------------------ colours
NAMED_COLOURS = {
    "white": (0.93, 0.93, 0.91), "grey": (0.60, 0.60, 0.60), "gray": (0.60, 0.60, 0.60),
    "lightgrey": (0.78, 0.78, 0.76), "lightgray": (0.78, 0.78, 0.76),
    "darkgrey": (0.35, 0.35, 0.35), "darkgray": (0.35, 0.35, 0.35), "black": (0.15, 0.15, 0.15),
    "red": (0.62, 0.22, 0.18), "brown": (0.45, 0.30, 0.20), "beige": (0.84, 0.78, 0.64),
    "tan": (0.78, 0.68, 0.52), "yellow": (0.88, 0.80, 0.45), "orange": (0.85, 0.52, 0.25),
    "blue": (0.30, 0.42, 0.62), "green": (0.32, 0.50, 0.32), "silver": (0.75, 0.75, 0.78),
    "maroon": (0.45, 0.15, 0.15), "cream": (0.92, 0.88, 0.76),
}


def parse_colour(value):
    """OSM colour tag ('#a0522d', 'a0522d', 'brown') -> (r, g, b) in 0..1, or None."""
    if not value:
        return None
    v = str(value).strip().lower().replace(" ", "")
    if v in NAMED_COLOURS:
        return NAMED_COLOURS[v]
    v = v.lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) == 6:
        try:
            return tuple(int(v[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        except ValueError:
            return None
    return None


# ------------------------------------------------------------------ PNG + textures
def write_png(path, rgb):
    """Write an (H, W, 3) uint8 array as a PNG using only zlib."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


# Default wall colour per style (used when the building has no building:colour tag).
STYLE_WALL = {
    "brick": (0.58, 0.27, 0.20),
    "siding": (0.86, 0.83, 0.74),
    "stucco": (0.82, 0.76, 0.64),
    "glass": (0.62, 0.66, 0.70),
    "metal": (0.66, 0.67, 0.68),
}
STYLE_ROOF = {
    "brick": (0.30, 0.30, 0.32),
    "siding": (0.33, 0.27, 0.24),
    "stucco": (0.55, 0.30, 0.22),
    "glass": (0.55, 0.56, 0.58),
    "metal": (0.72, 0.73, 0.74),
}


def facade_texture(style, wall_rgb, size=64):
    """
    One 'bay' of wall (about 3 m wide x 1 storey high): wall material plus one window.
    The mesh UVs repeat it along the wall and up each floor. Returns the PNG path (cached).
    The image is vertically symmetric, so it looks right whichever way the renderer flips v.
    """
    os.makedirs(TEXTURE_DIR, exist_ok=True)
    key = f"{style}_{int(wall_rgb[0] * 255):02x}{int(wall_rgb[1] * 255):02x}{int(wall_rgb[2] * 255):02x}"
    path = os.path.join(TEXTURE_DIR, f"facade_{key}.png")
    if os.path.exists(path):
        return path

    rng = np.random.default_rng(zlib.crc32(key.encode()))      # same texture every run
    n = size
    yy, xx = np.mgrid[0:n, 0:n]
    base = np.array(wall_rgb, dtype=float)
    img = np.ones((n, n, 3)) * base
    img *= (1.0 + rng.normal(0, 0.035, (n, n, 1)))                 # subtle grain

    if style == "brick":
        mortar = (yy % 8 == 0) | (((xx + 8 * ((yy // 8) % 2)) % 16) == 0)
        img[mortar] = base * 0.55 + 0.35
        brick_tone = rng.normal(0, 0.06, (n // 8 + 1, n // 16 + 2, 1))
        img *= 1.0 + brick_tone[yy // 8, (xx + 8 * ((yy // 8) % 2)) // 16]
        win = (xx >= 16) & (xx < 48) & (yy >= 14) & (yy < 50)
    elif style == "siding":
        img[(yy % 6) == 0] *= 0.82                                  # horizontal lap boards
        win = (xx >= 18) & (xx < 46) & (yy >= 16) & (yy < 48)
    elif style == "glass":
        win = (xx >= 3) & (xx < 61) & (yy >= 6) & (yy < 58)          # curtain wall
    elif style == "metal":
        img[(xx % 6) < 2] *= 0.85                                    # corrugated panels
        win = (xx >= 22) & (xx < 42) & (yy >= 26) & (yy < 38)        # small high window
    else:                                                            # stucco
        win = (xx >= 18) & (xx < 46) & (yy >= 16) & (yy < 48)

    # window: dark glass with a soft diagonal sky reflection, light frame around it
    frame = np.zeros_like(win)
    ys, xs = np.where(win)
    if len(ys):
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        frame = (yy >= y0 - 2) & (yy <= y1 + 2) & (xx >= x0 - 2) & (xx <= x1 + 2) & ~win
        refl = 0.5 + 0.5 * np.clip(1.0 - np.abs((xx - x0) - (yy - y0) * 0.6) / 25.0, 0, 1)
        glass = np.stack([0.20 + 0.18 * refl, 0.26 + 0.20 * refl, 0.32 + 0.24 * refl], axis=-1)
        img[win] = glass[win]
        img[frame] = np.array([0.88, 0.87, 0.84]) if style != "glass" else base * 0.7
    img = np.clip(img, 0, 1)
    img = (img + img[::-1]) / 2 if style != "brick" else img       # keep symmetric top/bottom
    write_png(path, (img * 255).astype(np.uint8))
    return path


# ------------------------------------------------------------------ geometry helpers
def simplify(ring, tol):
    """Douglas-Peucker simplification of a closed ring (keeps at least 3 points)."""
    if len(ring) <= 4 or tol <= 0:
        return list(ring)

    def dp(pts):
        if len(pts) < 3:
            return pts
        a, b = pts[0], pts[-1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy) or 1e-12
        dmax, idx = 0.0, 0
        for i in range(1, len(pts) - 1):
            d = abs(dy * (pts[i][0] - a[0]) - dx * (pts[i][1] - a[1])) / L
            if d > dmax:
                dmax, idx = d, i
        if dmax <= tol:
            return [a, b]
        return dp(pts[:idx + 1])[:-1] + dp(pts[idx:])

    # split the ring at its farthest point from the first point so both halves are open chains
    far = max(range(len(ring)), key=lambda i: math.hypot(ring[i][0] - ring[0][0], ring[i][1] - ring[0][1]))
    out = dp(list(ring[:far + 1]))[:-1] + dp(list(ring[far:]) + [ring[0]])[:-1]
    return out if len(out) >= 3 else list(ring)


class MeshBuilder:
    """Accumulates triangles (with normals and UVs) and splits them into PyBullet-sized chunks."""

    def __init__(self):
        self.chunks = []
        self._new()

    def _new(self):
        self.v, self.n, self.uv, self.i = [], [], [], []
        self.chunks.append((self.v, self.n, self.uv, self.i))

    def _room(self, nv, ni):
        if len(self.v) + nv > MAX_VERTS or len(self.i) + ni > MAX_INDICES:
            if self.v:
                self._new()

    def add(self, verts, tris, normal=None, normals=None, uvs=None):
        """verts: list of (x, y, z); tris: index triples into verts (counter-clockwise from outside)."""
        if not tris:
            return
        self._room(len(verts), 3 * len(tris))
        base = len(self.v)
        self.v.extend([list(map(float, p)) for p in verts])
        if normals is None:
            normals = [normal or (0.0, 0.0, 1.0)] * len(verts)
        self.n.extend([list(map(float, q)) for q in normals])
        self.uv.extend([list(map(float, q)) for q in (uvs or [(p[0], p[1]) for p in verts])])
        for a, b, c in tris:
            self.i.extend([base + a, base + b, base + c])

    def add_flat_polygon(self, poly, z, triangulate):
        tris = triangulate(poly)
        if tris:
            self.add([(x, y, z) for x, y in poly], tris)

    def add_strip(self, a, b, width, z, caps=(True, True)):
        """Flat quad of the given width from a to b, plus round caps so polylines join cleanly.
        caps: (at a, at b); leave a cap off at a dead end so roads don't end in a round blob."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            return
        nx, ny = -dy / L * width / 2, dx / L * width / 2
        quad = [(a[0] + nx, a[1] + ny, z), (a[0] - nx, a[1] - ny, z),
                (b[0] - nx, b[1] - ny, z), (b[0] + nx, b[1] + ny, z)]
        self.add(quad, [(0, 1, 2), (0, 2, 3)])          # (nx, ny) is the left normal -> faces up
        for c, cap in zip((a, b), caps):
            if cap:
                self.add_disc(c, width / 2, z)

    def add_disc(self, c, r, z, sides=8):
        verts = [(c[0], c[1], z)] + [(c[0] + r * math.cos(2 * math.pi * k / sides),
                                      c[1] + r * math.sin(2 * math.pi * k / sides), z) for k in range(sides)]
        self.add(verts, [(0, 1 + k, 1 + (k + 1) % sides) for k in range(sides)])

    def meshes(self):
        """Non-empty chunks as (vertices, normals, uvs, indices)."""
        return [c for c in self.chunks if c[0]]


def ensure_ccw(poly):
    area = 0.5 * sum(poly[i - 1][0] * poly[i][1] - poly[i][0] * poly[i - 1][1] for i in range(len(poly)))
    return list(poly) if area > 0 else list(reversed(poly))


def wall_mesh(poly, height, bay_w, storey_h):
    """Outward-facing walls with flat normals and UVs: u counts bays along the wall, v counts storeys."""
    poly = ensure_ccw(poly)
    verts, normals, uvs, tris = [], [], [], []
    u = 0.0
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        nrm = (dy / L, -dx / L, 0.0)                      # outward for a CCW ring
        u1 = u + L / bay_w
        v1 = height / storey_h
        base = len(verts)
        verts += [(a[0], a[1], 0.0), (b[0], b[1], 0.0), (b[0], b[1], height), (a[0], a[1], height)]
        normals += [nrm] * 4
        uvs += [(u, 0.0), (u1, 0.0), (u1, v1), (u, v1)]
        tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        u = u1 % 1.0                                      # keep UVs small; texture repeats per bay
    return verts, normals, uvs, tris


# ------------------------------------------------------------------ trees
def _icosphere():
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    v = [tuple(c / math.sqrt(sum(q * q for q in p)) for c in p) for p in v]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
         (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    cache = {}

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in cache:
            p = [(v[a][i] + v[b][i]) / 2 for i in range(3)]
            m = math.sqrt(sum(c * c for c in p))
            v.append(tuple(c / m for c in p))
            cache[key] = len(v) - 1
        return cache[key]

    out = []
    for a, b, c in f:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        out += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return v, out


_ICO = _icosphere()


def _flat(verts, tris):
    """Duplicate vertices per face so each triangle gets its own normal (faceted low-poly look)."""
    out_v, out_n, out_t = [], [], []
    for a, b, c in tris:
        pa, pb, pc = verts[a], verts[b], verts[c]
        u = [pb[i] - pa[i] for i in range(3)]
        w = [pc[i] - pa[i] for i in range(3)]
        n = (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])
        m = math.sqrt(sum(q * q for q in n)) or 1.0
        n = tuple(q / m for q in n)
        base = len(out_v)
        out_v += [pa, pb, pc]
        out_n += [n, n, n]
        out_t.append((base, base + 1, base + 2))
    return out_v, out_n, out_t


def tree_canopy_mesh(kind, seed, radius=0.78):
    """
    Canopy mesh centred on (0, 0, 0) that stays inside the 0.75 m collision sphere's footprint,
    so what you see matches what the sensors avoid. Returns (verts, normals, tris).
      deciduous: lumpy, slightly squashed sphere
      conifer:   three stacked 8-sided cones
    """
    rng = random.Random(seed)
    if kind == "conifer":
        verts, tris = [], []
        tiers = [(-0.75, 0.80, 0.80), (-0.25, 0.62, 0.70), (0.20, 0.42, 0.65)]   # (base z, radius, height)
        sides = 8
        for z0, r, h in tiers:
            base = len(verts)
            rot = rng.uniform(0, math.pi)
            verts.append((0.0, 0.0, z0 + h))
            for k in range(sides):
                a = rot + 2 * math.pi * k / sides
                verts.append((r * math.cos(a), r * math.sin(a), z0))
            verts.append((0.0, 0.0, z0))
            for k in range(sides):
                p, q = base + 1 + k, base + 1 + (k + 1) % sides
                tris.append((base, p, q))                       # side
                tris.append((base + sides + 1, q, p))           # underside
        return _flat(verts, tris)
    v, f = _ICO
    jit = [radius * rng.uniform(0.82, 1.0) for _ in v]
    verts = [(p[0] * j, p[1] * j, p[2] * j * 0.85) for p, j in zip(v, jit)]
    return _flat(verts, f)


def tree_colours(kind, seed):
    rng = random.Random(seed * 7 + 3)
    if kind == "conifer":
        g = (0.12 + rng.uniform(-0.03, 0.03), 0.32 + rng.uniform(-0.05, 0.04), 0.16 + rng.uniform(-0.03, 0.03))
    else:
        g = (0.22 + rng.uniform(-0.06, 0.06), 0.44 + rng.uniform(-0.07, 0.06), 0.16 + rng.uniform(-0.04, 0.04))
    trunk = (0.33 + rng.uniform(-0.04, 0.04), 0.23, 0.15)
    return list(g) + [1.0], list(trunk) + [1.0]


# ================================================================== detailed props (v3)
# Colour variants so neighbouring buildings / cars don't all look identical.
SIDING_VARIANTS = [(0.90, 0.89, 0.85), (0.86, 0.82, 0.71), (0.74, 0.77, 0.78), (0.68, 0.74, 0.80),
                   (0.70, 0.75, 0.64), (0.80, 0.72, 0.58)]
BRICK_VARIANTS = [(0.58, 0.27, 0.20), (0.52, 0.24, 0.19), (0.63, 0.33, 0.24)]
HOUSE_ROOF_VARIANTS = [(0.28, 0.27, 0.27), (0.36, 0.28, 0.24), (0.30, 0.33, 0.36), (0.42, 0.24, 0.20)]
CAR_COLOURS = [(0.85, 0.85, 0.86), (0.12, 0.12, 0.13), (0.55, 0.56, 0.58), (0.60, 0.10, 0.10),
               (0.15, 0.25, 0.45), (0.75, 0.75, 0.72), (0.30, 0.32, 0.30)]
CAR_GLASS = (0.16, 0.19, 0.23)


def pick(variants, x, y):
    """Deterministic choice from a list, based on a position (same building -> same colour)."""
    return variants[int(abs(x * 73.1 + y * 37.7) * 10) % len(variants)]


def _tri_normal(a, b, c):
    u = [b[i] - a[i] for i in range(3)]
    w = [c[i] - a[i] for i in range(3)]
    n = (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0])
    m = math.sqrt(sum(q * q for q in n))
    return (n[0] / m, n[1] / m, n[2] / m) if m > 1e-12 else None


def add_faceted(mb, tris, outward_from=None, up=False):
    """Add triangles given as ((x,y,z), (x,y,z), (x,y,z)) with flat normals.
    outward_from: interior point; each triangle is flipped to face away from it.
    up: flip every triangle to face upward (flat ground pieces / decks)."""
    verts, normals, idx = [], [], []
    for a, b, c in tris:
        n = _tri_normal(a, b, c)
        if n is None:
            continue
        flip = False
        if up:
            flip = n[2] < 0
        elif outward_from is not None:
            m = [(a[i] + b[i] + c[i]) / 3 - outward_from[i] for i in range(3)]
            flip = sum(n[i] * m[i] for i in range(3)) < 0
        if flip:
            b, c = c, b
            n = (-n[0], -n[1], -n[2])
        base = len(verts)
        verts += [a, b, c]
        normals += [n, n, n]
        idx.append((base, base + 1, base + 2))
    if idx:
        mb.add(verts, idx, normals=normals)


def convex_hull(pts):
    pts = sorted(set((round(p[0], 6), round(p[1], 6)) for p in pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def min_area_rect(poly):
    """Smallest rotated rectangle around a polygon: (centre, angle of long side, length, width).
    length >= width."""
    hull = convex_hull(poly)
    best = None
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        c, s = math.cos(ang), math.sin(ang)
        us = [p[0] * c + p[1] * s for p in hull]
        vs = [-p[0] * s + p[1] * c for p in hull]
        area = (max(us) - min(us)) * (max(vs) - min(vs))
        if best is None or area < best[0]:
            best = (area, ang, min(us), max(us), min(vs), max(vs))
    if best is None:
        return None
    _, ang, u0, u1, v0, v1 = best
    L, W = u1 - u0, v1 - v0
    uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
    c, s = math.cos(ang), math.sin(ang)
    centre = (uc * c - vc * s, uc * s + vc * c)
    if W > L:
        L, W, ang = W, L, ang + math.pi / 2
    return centre, ang, L, W


def rect_corners(centre, ang, L, W):
    c, s = math.cos(ang), math.sin(ang)
    out = []
    for u, v in ((-L / 2, -W / 2), (L / 2, -W / 2), (L / 2, W / 2), (-L / 2, W / 2)):
        out.append((centre[0] + u * c - v * s, centre[1] + u * s + v * c))
    return out                                                     # counter-clockwise


def _local(centre, ang):
    c, s = math.cos(ang), math.sin(ang)
    return lambda u, v, z: (centre[0] + u * c - v * s, centre[1] + u * s + v * c, z)


def pitched_roof(rect, z0, z1, shape, overhang=0.0):
    """Roof on a rectangle. shape: 'gabled' | 'hipped' | 'pyramidal' | 'skillion'.
    Returns (roof_tris, gable_tris): gable ends are wall-coloured triangles (gabled only).
    The roof stays inside the rectangle's vertical extent [z0, z1]."""
    centre, ang, L, W = rect
    P = _local(centre, ang)
    e = overhang
    hl, hw = L / 2 + e, W / 2 + e
    slope = (z1 - z0) / max(W / 2, 1e-6)
    ze = z0 - e * slope * 0.5                                    # eave dips slightly with overhang
    ze = max(ze, z0 - 0.1)
    roof, gables = [], []
    if shape == "skillion":
        a, b, c, d = P(-hl, -hw, ze), P(hl, -hw, ze), P(hl, hw, z1), P(-hl, hw, z1)
        roof += [(a, b, c), (a, c, d)]
        gables += [(P(-L / 2, -W / 2, z0), P(-L / 2, W / 2, z0), P(-L / 2, W / 2, z1)),
                   (P(L / 2, -W / 2, z0), P(L / 2, W / 2, z1), P(L / 2, W / 2, z0))]
        gables += [(P(-L / 2, W / 2, z0), P(L / 2, W / 2, z0), P(L / 2, W / 2, z1)),
                   (P(-L / 2, W / 2, z0), P(L / 2, W / 2, z1), P(-L / 2, W / 2, z1))]
        return roof, gables
    if shape == "pyramidal":
        apex = P(0, 0, z1)
        cs = [P(-hl, -hw, ze), P(hl, -hw, ze), P(hl, hw, ze), P(-hl, hw, ze)]
        for i in range(4):
            roof.append((cs[i], cs[(i + 1) % 4], apex))
        return roof, gables
    if shape == "hipped":
        r = max(0.0, L / 2 - W / 2)                              # ridge half-length
        r1, r2 = P(-r, 0, z1), P(r, 0, z1)
        a, b, c, d = P(-hl, -hw, ze), P(hl, -hw, ze), P(hl, hw, ze), P(-hl, hw, ze)
        roof += [(a, b, r2), (a, r2, r1), (c, d, r1), (c, r1, r2), (b, c, r2), (d, a, r1)]
        return roof, gables
    # gabled (default)
    r1, r2 = P(-hl, 0, z1), P(hl, 0, z1)
    a, b, c, d = P(-hl, -hw, ze), P(hl, -hw, ze), P(hl, hw, ze), P(-hl, hw, ze)
    roof += [(a, b, r2), (a, r2, r1), (c, d, r1), (c, r1, r2)]
    gables += [(P(-L / 2, -W / 2, z0), P(-L / 2, W / 2, z0), P(-L / 2, 0, z1)),
               (P(L / 2, -W / 2, z0), P(L / 2, 0, z1), P(L / 2, W / 2, z0))]
    return roof, gables


def pyramid_on_polygon(poly, z0, z1):
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    apex = (cx, cy, z1)
    return [((poly[i][0], poly[i][1], z0), (poly[(i + 1) % len(poly)][0], poly[(i + 1) % len(poly)][1], z0), apex)
            for i in range(len(poly))]


def dome(centre, radius, z0, height, rings=6, segs=16):
    tris = []
    pts = []
    for r in range(rings + 1):
        phi = (math.pi / 2) * r / rings
        rr, zz = radius * math.cos(phi), z0 + height * math.sin(phi)
        pts.append([(centre[0] + rr * math.cos(2 * math.pi * k / segs),
                     centre[1] + rr * math.sin(2 * math.pi * k / segs), zz) for k in range(segs)])
    for r in range(rings):
        for k in range(segs):
            a, b = pts[r][k], pts[r][(k + 1) % segs]
            c, d = pts[r + 1][(k + 1) % segs], pts[r + 1][k]
            tris += [(a, b, c), (a, c, d)]
    return tris


def box_tris(centre, ang, L, W, z0, z1, bottom=False):
    """Rotated box (no bottom face unless asked) as triangles; orient with outward_from=centre."""
    P = _local(centre, ang)
    lo = [P(-L / 2, -W / 2, z0), P(L / 2, -W / 2, z0), P(L / 2, W / 2, z0), P(-L / 2, W / 2, z0)]
    hi = [P(-L / 2, -W / 2, z1), P(L / 2, -W / 2, z1), P(L / 2, W / 2, z1), P(-L / 2, W / 2, z1)]
    tris = [(hi[0], hi[1], hi[2]), (hi[0], hi[2], hi[3])]
    for i in range(4):
        j = (i + 1) % 4
        tris += [(lo[i], lo[j], hi[j]), (lo[i], hi[j], hi[i])]
    if bottom:
        tris += [(lo[0], lo[2], lo[1]), (lo[0], lo[3], lo[2])]
    return tris


def car_tris(centre, ang, scale):
    """Low-poly parked car (real size ~4.5 x 1.8 x 1.45 m), scaled. Returns (body_tris, glass_tris)."""
    s = scale
    body = box_tris(centre, ang, 4.5 * s, 1.8 * s, 0.25 * s, 0.95 * s)
    c, sn = math.cos(ang), math.sin(ang)
    cab_c = (centre[0] - 0.25 * s * c, centre[1] - 0.25 * s * sn)
    glass = box_tris(cab_c, ang, 2.3 * s, 1.62 * s, 0.95 * s, 1.45 * s)
    return body, glass


def vertical_strip_tris(a, b, z0, z1):
    """Thin wall/fence: one quad visible from both sides."""
    A0, B0, B1, A1 = (a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)
    return [(A0, B0, B1), (A0, B1, A1), (A0, B1, B0), (A0, A1, B1)]


def wall_band(poly, z0, z1, bay_w, storey_h):
    """Like wall_mesh, but from z0 to z1 (for building parts that start above the ground).
    UV v counts storeys from the ground so windows line up between parts."""
    poly = ensure_ccw(poly)
    verts, normals, uvs, tris = [], [], [], []
    u = 0.0
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        nrm = (dy / L, -dx / L, 0.0)
        u1 = u + L / bay_w
        base = len(verts)
        verts += [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)]
        normals += [nrm] * 4
        uvs += [(u, z0 / storey_h), (u1, z0 / storey_h), (u1, z1 / storey_h), (u, z1 / storey_h)]
        tris += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        u = u1 % 1.0
    return verts, normals, uvs, tris


def inner_parapet_tris(poly, z0, z1):
    """Inward-facing wall band, so a flat roof deck below the wall top shows a parapet from above."""
    poly = ensure_ccw(poly)
    tris = []
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        A0, B0, B1, A1 = (a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)
        tris += [(A0, B1, B0), (A0, A1, B1)]                  # reversed winding = faces inward
    return tris
