"""
Dock the operator panel (a Tk window) beside PyBullet's 3D window.

PyBullet's window is a native OpenGL window, so it can't be embedded in Tk; instead the two
windows are kept side by side:

  * start-up: the 3D window is resized to fill its monitor's work area next to the panel
  * the panel sits flush against the 3D window's right edge (left edge if there's no room on
    the right; overlapping it if there's no room on either side), top edges aligned
  * when the 3D window is moved or resized, the panel follows once the window stops moving
    (no chasing while it is being dragged)
  * whenever the 3D window is brought to the front, the panel is raised too (without taking
    the keyboard focus), so it can never end up hidden behind it
  * minimising the 3D window hides the panel; restoring it brings the panel back

Exact docking needs the Win32 API (ctypes, no extra packages). Every position comes from
GetWindowRect / SetWindowPos / GetMonitorInfo, which all use the same coordinates even with
display scaling (125 %, 150 %...). DWM frame bounds are NOT used: they are always in physical
pixels, which put the panel in the wrong place on scaled displays.
On macOS / Linux PyBullet's window can't be moved, so the panel just stays on top at the
right edge of the screen (see operator_panel.py).

The layout logic (dock_layout, Docker) only talks to a small "api" object, so it is tested with
a fake window system; Win32Api is the real one.
"""
import time

SETTLE_S = 0.15            # 3D window must be still this long before the panel follows it
MIN_SIM_W, MIN_SIM_H = 640, 480


# ------------------------------------------------------------------ layout --
def dock_layout(sim, panel_w, work, border, panel_h=0):
    """
    Where the panel goes, from outer window rects (l, t, r, b) as GetWindowRect reports them.
    border: width of the invisible resize border Windows 10/11 draws on the left, right and
    bottom of every window (0 on older Windows). Visible edges are flush, so the invisible
    borders overlap. Returns (x, y, side) for the panel's outer top-left; side is
    "right", "left" or "overlap". The panel is top-aligned with the 3D window, but kept
    inside the work area (never under the taskbar or off the top).
    """
    sl, st, sr, sb = sim
    wl, wt, wr, wb = work
    st = max(wt, min(st, wb + border - panel_h))
    vis_right, vis_left = sr - border, sl + border           # 3D window's visible edges
    if vis_right + panel_w - 2 * border <= wr + border:      # room on the right
        return vis_right - border, st, "right"
    if vis_left - (panel_w - 2 * border) >= wl - border:     # room on the left
        return vis_left - panel_w + border, st, "left"
    # no room either side (e.g. maximised 3D window): overlap its right edge, kept on top
    x = min(max(wl, vis_right - panel_w + border), wr - panel_w + border)
    return x, st, "overlap"


def start_layout(work, panel_w, border):
    """Outer rect for the 3D window at start-up: fill the work area beside the panel.
    None if the screen is too small (then the 3D window keeps its own size)."""
    wl, wt, wr, wb = work
    vis_w = (wr - wl) - (panel_w - 2 * border)
    vis_h = wb - wt
    if vis_w < MIN_SIM_W or vis_h < MIN_SIM_H:
        return None
    return (wl - border, wt, wl + vis_w + border, wb + border)


class Docker:
    """Keeps the panel docked; call poll() regularly (main loop, ~20 Hz)."""

    def __init__(self, api, sim, panel, panel_size, show_panel, hide_panel, clock=time.monotonic):
        self.api, self.sim, self.panel = api, sim, panel
        self.panel_size = panel_size        # () -> (outer width, outer height) of the panel
        self.show_panel, self.hide_panel = show_panel, hide_panel
        self.clock = clock
        self.side = None
        self._last_rect = None
        self._changed_at = None
        self._last_fg = None
        self._hidden = False

    def start(self, resize_sim=True):
        """First placement: size the 3D window to share its monitor with the panel, dock."""
        border = self.api.border(self.sim)
        if resize_sim and not self.api.is_maximized(self.sim):
            r = start_layout(self.api.work_area(self.sim), self.panel_size()[0], border)
            if r is not None:
                self.api.set_rect(self.sim, r)
        self.place()
        self._last_rect = self.api.rect(self.sim)
        return self.side

    def place(self):
        sim = self.api.rect(self.sim)
        pw, ph = self.panel_size()
        x, y, self.side = dock_layout(sim, pw, self.api.work_area(self.sim),
                                      self.api.border(self.sim), ph)
        self.api.move(self.panel, x, y)
        self.api.raise_above(self.panel)

    def poll(self):
        api, now = self.api, self.clock()
        if not api.exists(self.sim):
            return
        if api.is_minimized(self.sim):                      # follow minimise / restore
            if not self._hidden:
                self._hidden = True
                self.hide_panel()
            return
        if self._hidden:
            self._hidden = False
            self.show_panel()
            self.place()
        rect = api.rect(self.sim)
        if rect != self._last_rect:                         # moving / resizing: wait
            self._last_rect, self._changed_at = rect, now
        elif self._changed_at is not None and now - self._changed_at >= SETTLE_S:
            self._changed_at = None                         # stopped: snap the panel to it
            self.place()
        fg = api.foreground()                               # 3D window brought to front:
        if fg != self._last_fg:                             # raise the panel with it
            self._last_fg = fg
            if fg == self.sim:
                api.raise_above(self.panel)


# ------------------------------------------------------------------- Win32 --
class Win32Api:
    """The Win32 calls Docker needs (ctypes only)."""

    def __init__(self):
        import ctypes
        import sys
        from ctypes import wintypes
        self.c, self.w = ctypes, wintypes
        self.u = ctypes.windll.user32
        self.u.GetForegroundWindow.restype = wintypes.HWND
        self.win10 = sys.getwindowsversion().major >= 10

    def _rect(self, h, fn):
        r = self.w.RECT()
        fn(self.w.HWND(h), self.c.byref(r))
        return r.left, r.top, r.right, r.bottom

    def rect(self, h):
        return self._rect(h, self.u.GetWindowRect)

    def exists(self, h):
        return bool(self.u.IsWindow(self.w.HWND(h)))

    def is_minimized(self, h):
        return bool(self.u.IsIconic(self.w.HWND(h)))

    def is_maximized(self, h):
        return bool(self.u.IsZoomed(self.w.HWND(h)))

    def border(self, h):
        """Invisible resize border (Windows 10/11 only; part of the visible frame before)."""
        if not self.win10:
            return 0
        style = self.u.GetWindowLongW(self.w.HWND(h), -16)          # GWL_STYLE
        if not style & 0x00040000:                                  # WS_THICKFRAME
            return 0
        b = self.u.GetSystemMetrics(32) + self.u.GetSystemMetrics(92)   # SIZEFRAME + PADDED
        return max(0, b - 1)                                        # 1 px is visible

    def work_area(self, h):
        """Work area (screen minus taskbar) of the monitor the window is on."""
        class MONITORINFO(self.c.Structure):
            _fields_ = [("cbSize", self.w.DWORD), ("rcMonitor", self.w.RECT),
                        ("rcWork", self.w.RECT), ("dwFlags", self.w.DWORD)]
        mon = self.u.MonitorFromWindow(self.w.HWND(h), 2)           # MONITOR_DEFAULTTONEAREST
        mi = MONITORINFO()
        mi.cbSize = self.c.sizeof(MONITORINFO)
        if mon and self.u.GetMonitorInfoW(mon, self.c.byref(mi)):
            r = mi.rcWork
            return r.left, r.top, r.right, r.bottom
        r = self.w.RECT()
        self.u.SystemParametersInfoW(0x30, 0, self.c.byref(r), 0)  # SPI_GETWORKAREA
        return r.left, r.top, r.right, r.bottom

    def set_rect(self, h, r):
        # SWP_NOZORDER | SWP_NOACTIVATE
        self.u.SetWindowPos(self.w.HWND(h), None, r[0], r[1], r[2] - r[0], r[3] - r[1], 0x4 | 0x10)

    def move(self, h, x, y):
        # SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE
        self.u.SetWindowPos(self.w.HWND(h), None, int(x), int(y), 0, 0, 0x1 | 0x4 | 0x10)

    def raise_above(self, h):
        """Top of the normal (non-topmost) windows, without activating it."""
        # HWND_TOP; SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE
        self.u.SetWindowPos(self.w.HWND(h), self.w.HWND(0), 0, 0, 0, 0, 0x1 | 0x2 | 0x10)

    def foreground(self):
        h = self.u.GetForegroundWindow()
        return int(h) if h else None

    def find_sim_window(self, exclude=()):
        """Largest visible top-level window of this process that isn't a Tk window: PyBullet's."""
        import os
        found, pid = [], os.getpid()
        proc = self.c.WINFUNCTYPE(self.w.BOOL, self.w.HWND, self.w.LPARAM)

        def cb(h, _l):
            owner = self.w.DWORD()
            self.u.GetWindowThreadProcessId(h, self.c.byref(owner))
            if owner.value == pid and self.u.IsWindowVisible(h):
                buf = self.c.create_unicode_buffer(256)
                self.u.GetClassNameW(h, buf, 256)
                hv = int(h)
                if hv not in exclude and not buf.value.startswith("Tk"):
                    l, t, r, b = self.rect(hv)
                    found.append(((r - l) * (b - t), hv))
            return True
        self.u.EnumWindows(proc(cb), 0)
        return max(found)[1] if found else None


def win32_api():
    """Win32Api on Windows, else None."""
    import sys
    if not sys.platform.startswith("win"):
        return None
    try:
        return Win32Api()
    except Exception:
        return None
