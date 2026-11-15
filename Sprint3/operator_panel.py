"""
Operator control panel (PBI-09) - a small Tkinter window that runs next to the PyBullet view.

PyBullet's built-in panel only has plain sliders and buttons (no text labels, no colours), so
the live operator controls live here instead:

    * plain-text section headers
    * zone buttons laid out like the map (B north-west, A north-east, C south-west, D south-east),
      coloured by zone status, with the selected zone outlined
    * colour-coded command buttons that name the selected zone, plus "Return to original plan"
    * the latest message (green accepted / red rejected / blue mission event) and a scrolling,
      time-stamped log of every mission message: operator commands and their replies, mission
      events (started, zone searched, person found) and messages set by main.py (setup accepted,
      navigation aborted, returned to launch, plan complete)
    * plan / searched / excluded summary, heatmap and camera buttons

The window never blocks the simulation: main.py calls poll() from its loop (about 20 times a
second), which processes Tk events and returns what the operator pressed since the last call.
The panel has no mission logic of its own: commands are validated by operator_guidance.Mission.
"""

ZONE_FILL = {"next": "#2e7d32", "planned": "#1e88e5", "searched": "#757575", "excluded": "#c62828"}
STATUS_TEXT = {"next": "NEXT", "planned": "PLANNED", "searched": "SEARCHED", "excluded": "EXCLUDED"}
CMD_COLOURS = {                 # Cmd name -> (background, background while pressed)
    "INVESTIGATE": ("#ef6c00", "#b53d00"),
    "SEARCH_FIRST": ("#2e7d32", "#1b5e20"),
    "AVOID": ("#c62828", "#8e0000"),
    "REINCLUDE": ("#1565c0", "#0d47a1"),
    "RETURN_PLAN": ("#5e35b1", "#311b92"),
}
PANEL_BG = "#f4f4f4"
SCREEN_MARGIN = 90             # px kept free for the title bar and taskbar
FULL_LAYOUT_MIN_H = 980        # screens shorter than this (available px) get the compact layout
MIN_TEXT_SCALE = 0.75          # still too tall: shrink all text (and so buttons/tiles) down to 75%
MIN_FONT_PT = 8                # ...but never below this (smaller is unreadable); then it scrolls
LOG_LINES = 7                   # visible lines in the message log (it scrolls)
LOG_KEEP = 300                  # older messages are dropped past this many
OK_FG, BAD_FG, SUB_FG, EVENT_FG = "#1b5e20", "#b71c1c", "#555555", "#0d47a1"
LATEST_BG = {"ok": "#e8f5e9", "bad": "#ffebee", "event": "#e3f2fd", "info": "white"}
FLASH_MS = 350                  # how long a pressed button stays dark


def _grid_slot(xy):
    """Row/column of a zone in the 2x2 map-shaped grid (north on top, west on the left)."""
    return (0 if xy[1] >= 0 else 1), (0 if xy[0] < 0 else 1)


class OperatorPanel:
    def __init__(self, zones, commands, zone_commands, directions=None, camera_modes=None,
                 title="Drone Search & Rescue - Operator Controls"):
        import tkinter as tk
        from tkinter import font as tkfont
        self.tk = tk
        self.zones = dict(zones)
        self.commands = list(commands)
        self.zone_commands = set(zone_commands)
        self.events = []
        self.closed = False
        self._shown = {}            # widget -> last text/colour, so refresh() only touches changes
        self._hist_n = 0            # mission.history entries already copied into the log
        self._last_resp = None      # last mission.response seen (catches messages not in history)
        self._latest = None         # (text, kind) of the newest message
        self._resized = False       # sized again once the first refresh filled in the labels
        self.docker = None          # window_dock.Docker while docked to the 3D window (Windows)

        root = self.root = tk.Tk()
        root.title(title)
        root.configure(bg=PANEL_BG)
        root.resizable(False, False)                 # sizes itself to its content / the screen
        root.protocol("WM_DELETE_WINDOW", self._ask_close)
        # Fit the screen: compact layout on short screens, plus a scrollbar if still too tall.
        self.avail_h = root.winfo_screenheight() - SCREEN_MARGIN
        self.compact = c = self.avail_h < FULL_LAYOUT_MIN_H
        try:
            root.attributes("-topmost", True)       # stay visible above the 3D window
        except tk.TclError:
            pass
        base = tkfont.nametofont("TkDefaultFont")
        size = max(10, base.cget("size")) - (1 if c else 0)
        self.f_head = base.copy(); self.f_head.configure(size=size + 3, weight="bold")
        self.f_bold = base.copy(); self.f_bold.configure(size=size + 1, weight="bold")
        self.f_norm = base.copy(); self.f_norm.configure(size=size)
        self.f_small = base.copy(); self.f_small.configure(size=max(8, size - 1))
        self.fonts = [self.f_head, self.f_bold, self.f_norm, self.f_small]
        self.font_sizes = [f.cget("size") for f in self.fonts]   # full-size values (points)
        self.text_scale = 1.0

        self.btn_pady = 3 if c else 6
        # content lives in a canvas so it can scroll when the window is shorter than it
        self.canvas = tk.Canvas(root, bg=PANEL_BG, highlightthickness=0, bd=0)
        self.vsb = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        frm = self.content = tk.Frame(self.canvas, bg=PANEL_BG, padx=12 if c else 14,
                                      pady=6 if c else 10)
        self.canvas.create_window(0, 0, window=frm, anchor="nw")
        frm.bind("<Configure>", lambda _e: self._fit_scroll())
        self.canvas.bind("<Configure>", lambda _e: self._fit_scroll())
        root.bind_all("<MouseWheel>", self._wheel)          # Windows / macOS
        root.bind_all("<Button-4>", self._wheel)            # Linux
        root.bind_all("<Button-5>", self._wheel)
        row = 0

        def header(text):
            nonlocal row
            tk.Label(frm, text=text, font=self.f_head, bg=PANEL_BG, anchor="w").grid(
                row=row, column=0, sticky="w", pady=(5, 2) if c else (10, 4))
            row += 1

        # --- mission
        header("Mission")
        top = tk.Frame(frm, bg=PANEL_BG); top.grid(row=row, column=0, sticky="we"); row += 1
        self.start_btn = self._button(top, "START MISSION", ("#2e7d32", "#1b5e20"),
                                      lambda: self._emit(("start",)), width=16)
        self.start_btn.pack(side="left")
        self.mission_lbl = tk.Label(top, text="", font=self.f_bold, bg=PANEL_BG, padx=10)
        self.mission_lbl.pack(side="left")

        # --- zones, laid out like the map
        header("1. Select a zone")
        grid = tk.Frame(frm, bg=PANEL_BG); grid.grid(row=row, column=0, sticky="w"); row += 1
        self.zone_frames, self.zone_lbls = {}, {}
        for z, xy in self.zones.items():
            r, col = _grid_slot(xy)
            outline = tk.Frame(grid, bg=PANEL_BG, padx=4, pady=4)     # black when selected
            outline.grid(row=r, column=col, padx=3, pady=2 if self.compact else 3)
            d = (directions or {}).get(z, "")
            lbl = tk.Label(outline, text="", font=self.f_bold, fg="white", width=16,
                           height=3 if self.compact else 4,
                           cursor="hand2", relief="raised", bd=2, justify="center")
            lbl.pack()
            lbl.bind("<Button-1>", lambda _e, z=z: self._emit(("select", z)))
            lbl._direction = d
            self.zone_frames[z], self.zone_lbls[z] = outline, lbl
        tk.Label(frm, text="Map view: north is up. Colour = status.", font=self.f_small,
                 fg=SUB_FG, bg=PANEL_BG).grid(row=row, column=0, sticky="w"); row += 1
        legend = tk.Frame(frm, bg=PANEL_BG); legend.grid(row=row, column=0, sticky="w"); row += 1
        for st in ("next", "planned", "searched", "excluded"):
            tk.Label(legend, text=f" {STATUS_TEXT[st].title()} ", bg=ZONE_FILL[st], fg="white",
                     font=self.f_small).pack(side="left", padx=(0, 4))

        # --- commands
        self.cmd_head = tk.Label(frm, text="", font=self.f_head, bg=PANEL_BG, anchor="w")
        self.cmd_head.grid(row=row, column=0, sticky="w", pady=(12, 4)); row += 1
        cmds = tk.Frame(frm, bg=PANEL_BG); cmds.grid(row=row, column=0, sticky="we"); row += 1
        self.cmd_btns = {}
        for i, cmd in enumerate(self.commands):
            colours = CMD_COLOURS.get(cmd.name, ("#455a64", "#263238"))
            b = self._button(cmds, "", colours, lambda cmd=cmd: self._emit(("cmd", cmd)), width=34)
            b.grid(row=i, column=0, sticky="we", pady=1 if c else 2)
            self.cmd_btns[cmd] = b

        # --- latest message + status + plan
        header("Mission messages")
        self.result_lbl = tk.Label(frm, text="No messages yet.", font=self.f_bold, bg="white",
                                   fg=SUB_FG, anchor="w", justify="left", wraplength=330,
                                   padx=8, pady=6, relief="groove", bd=2, width=40)
        self.result_lbl.grid(row=row, column=0, sticky="we"); row += 1
        # full log, newest at the bottom; read-only, scrolls
        logf = tk.Frame(frm, bg=PANEL_BG); logf.grid(row=row, column=0, sticky="we", pady=(6, 0)); row += 1
        self.log = tk.Text(logf, height=4 if c else LOG_LINES, width=46, wrap="word", font=self.f_small,
                           bg="white", relief="sunken", bd=1, padx=4, pady=2, cursor="arrow")
        sb = tk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set, state="disabled")
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.log.tag_configure("time", foreground="#888888")
        for kind, colour in (("ok", OK_FG), ("bad", BAD_FG), ("event", EVENT_FG), ("info", "#222222")):
            self.log.tag_configure(kind, foreground=colour)
        self.status_lbl = tk.Label(frm, text="", font=self.f_norm, bg=PANEL_BG, anchor="w",
                                   justify="left", wraplength=340)
        self.status_lbl.grid(row=row, column=0, sticky="w", pady=(6, 0)); row += 1
        self.plan_lbl = tk.Label(frm, text="", font=self.f_norm, bg=PANEL_BG, anchor="w",
                                 justify="left", wraplength=340)
        self.plan_lbl.grid(row=row, column=0, sticky="w"); row += 1

        # --- view
        header("View")
        view = tk.Frame(frm, bg=PANEL_BG); view.grid(row=row, column=0, sticky="w"); row += 1
        grey = ("#546e7a", "#37474f")
        self._button(view, "Show / hide heatmap", grey, lambda: self._emit(("heat",)),
                     width=18).pack(side="left", padx=(0, 6))
        self.cam_btn = self._button(view, "Camera", grey, lambda: self._emit(("cam",)), width=18)
        self.cam_btn.pack(side="left")
        tk.Label(frm, text="Free camera: click the 3D window first, then use the arrow keys.",
                 font=self.f_small, fg=SUB_FG, bg=PANEL_BG).grid(row=row, column=0, sticky="w",
                                                                pady=(6, 0)); row += 1
        self._size_to_screen()
        root.update()

    # ------------------------------------------------------------------ widgets --
    def _button(self, parent, text, colours, action, width=20):
        """A coloured click target. (tk.Button ignores bg colours on macOS; a Label does not.)"""
        tk = self.tk
        bg, dark = colours
        b = tk.Label(parent, text=text, bg=bg, fg="white", font=self.f_bold, width=width,
                     padx=6, pady=self.btn_pady, relief="raised", bd=2, cursor="hand2")
        b._colours = colours

        def press(_e):
            b.configure(bg=dark, relief="sunken")
            action()
            b.after(FLASH_MS, lambda: b.winfo_exists() and b.configure(bg=b._colours[0],
                                                                       relief="raised"))
        b.bind("<Button-1>", press)
        return b

    def _set(self, widget, **kw):
        """configure() only when something changed (keeps poll() cheap)."""
        key = tuple(sorted(kw.items()))
        if self._shown.get(widget) != key:
            self._shown[widget] = key
            widget.configure(**kw)

    def _emit(self, event):
        self.events.append(event)

    def _on_close(self):
        self.closed = True
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass

    def _ask_close(self):
        """Window close button: confirm, because closing the panel ends the simulation."""
        from tkinter import messagebox
        try:
            ok = messagebox.askyesno(
                "End the mission?",
                "Closing the operator panel ends the simulation.\n\nAre you sure you want to close it?",
                icon="warning", default="no", parent=self.root)
        except self.tk.TclError:
            ok = True
        if ok:
            self._on_close()

    # ------------------------------------------------------------ fit to screen --
    def _shrink_text_to_fit(self):
        """If the content is taller than the screen, scale every font down (buttons, tiles and
        labels are sized in characters, so they shrink with it), but not below MIN_TEXT_SCALE."""
        root = self.root
        for _ in range(8):                               # usually 1-3 passes
            root.update_idletasks()
            h = self.content.winfo_reqheight()
            if h <= self.avail_h or self.text_scale <= MIN_TEXT_SCALE:
                return
            # text is only part of the height (padding doesn't shrink), so aim a bit lower
            target = max(MIN_TEXT_SCALE, min(self.text_scale * (self.avail_h / h) ** 1.4,
                                             self.text_scale - 0.04))   # fonts step in whole points
            self.text_scale = round(target, 3)
            for f, full in zip(self.fonts, self.font_sizes):
                sign = -1 if full < 0 else 1                 # negative size = pixels in Tk
                f.configure(size=sign * max(MIN_FONT_PT, round(abs(full) * self.text_scale)))
            self._rewrap()

    def _rewrap(self):
        """Wrap long messages to the width the message box really has (it changes with the font)."""
        self.root.update_idletasks()
        box = self.result_lbl.winfo_reqwidth() - 2 * 8 - 8         # minus padding and border
        self.result_lbl.configure(wraplength=max(150, box))
        for lbl in (self.status_lbl, self.plan_lbl):
            lbl.configure(wraplength=max(150, box + 10))

    def _size_to_screen(self):
        """Window as tall as its content, but never taller than the screen (then it scrolls).
        Placed at the top-right of the screen so the whole height is on screen."""
        root = self.root
        self._shrink_text_to_fit()
        self._rewrap()
        root.update_idletasks()
        w, h = self.content.winfo_reqwidth(), self.content.winfo_reqheight()
        h_win = min(h, self.avail_h)
        root.maxsize(10000, max(h, 200))             # can't be stretched past its content
        self.canvas.configure(width=w, height=h_win)
        root.geometry("")                            # let the window follow the canvas size
        root.update_idletasks()
        if self.docker is not None:                  # docked: the docker decides the position
            try:
                self.docker.place()
            except Exception:
                self.docker = None
        if self.docker is None:
            x = max(0, root.winfo_screenwidth() - w - 40)
            root.geometry(f"+{x}+0")
        self._fit_scroll()

    def _fit_scroll(self):
        """Keep the window as tall as its content (up to the screen height) and show the
        scrollbar only when the content doesn't fit. Runs whenever the content changes size."""
        if getattr(self, "_fitting", False):            # shrinking fonts re-triggers this
            return
        try:
            self._fitting = True
            if self.content.winfo_reqheight() > self.avail_h and hasattr(self, "plan_lbl"):
                self._shrink_text_to_fit()             # content grew past the screen: shrink text
            req = self.content.winfo_reqheight()
            want = min(req, self.avail_h)
            if self.canvas.winfo_pixels(self.canvas.cget("height")) != want:
                self.root.maxsize(10000, max(req, 200))
                self.canvas.configure(height=want)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            need = req > want + 1
            shown = bool(self.vsb.winfo_ismapped())
            if need and not shown:
                self.vsb.pack(side="right", fill="y", before=self.canvas)
            elif not need and shown:
                self.vsb.pack_forget()
                self.canvas.yview_moveto(0)
        except self.tk.TclError:
            pass
        finally:
            self._fitting = False

    def _wheel(self, event):
        """Mouse wheel scrolls the panel (the message log scrolls itself)."""
        if event.widget is self.log or not self.vsb.winfo_ismapped():
            return
        if getattr(event, "num", None) == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        else:
            self.canvas.yview_scroll(1, "units")

    # ---------------------------------------------------------------- messages --
    @staticmethod
    def _kind(source, msg):
        """Colour class of a message: ok (accepted command), bad (rejected / aborted),
        event (mission events: start, zone searched, found, ...), info (everything else)."""
        low = msg.lower()
        if low.startswith("rejected") or "abort" in low:
            return "bad"
        if low.startswith("received"):
            return "ok"
        if source in ("START", "DETECT", "ARRIVED", "MISSION") or "found" in low or "searched" in low:
            return "event"
        return "info"

    def _add_message(self, ts, source, msg):
        import time as _time
        kind = self._kind(source, msg)
        stamp = _time.strftime("%H:%M:%S", _time.localtime(ts))
        log = self.log
        log.configure(state="normal")
        if float(log.index("end-1c")) > 1.0:            # not the first line
            log.insert("end", "\n")
        log.insert("end", stamp + "  ", "time")
        log.insert("end", msg, kind)
        extra = int(float(log.index("end-1c"))) - LOG_KEEP
        if extra > 0:
            log.delete("1.0", f"{extra + 1}.0")
        log.configure(state="disabled")
        log.see("end")
        self._latest = (msg, kind)

    def _collect_messages(self, mission):
        """Copy new mission messages into the log: every mission.history entry, plus any
        mission.response that main.py set directly (those don't go through history)."""
        import time as _time
        hist = mission.history
        # history_count = lines ever recorded (the list itself is capped, PBI-10), so new lines
        # are still found once the cap is reached.
        total = getattr(mission, "history_count", len(hist))
        if self._hist_n > total:                        # history was reset (e.g. restored)
            self._hist_n = 0
        new = min(total - self._hist_n, len(hist))
        for ts, source, msg in (hist[-new:] if new > 0 else []):
            self._add_message(ts, str(source), msg)
        logged_last = hist[-1][2] if hist else None
        self._hist_n = total
        # Zone arrivals are logged by Mission.arrived() (PBI-10), so they come in with history.
        resp = mission.response
        if resp and resp != self._last_resp and resp != logged_last:
            self._add_message(_time.time(), "MISSION", resp)
        self._last_resp = resp

    # ----------------------------------------------------------------- docking --
    def dock(self):
        """Dock beside PyBullet's 3D window (see window_dock.py). Windows only; elsewhere the
        panel stays on top at the right edge of the screen. Returns a short description."""
        import window_dock
        api = window_dock.win32_api()
        if api is None:
            return "kept on top at the right edge of the screen (docking needs Windows)"
        try:
            self.root.update_idletasks()
            me = int(self.root.wm_frame(), 16)
            sim = api.find_sim_window(exclude={me})
            if sim is None:
                return "not docked: the 3D window wasn't found (panel kept on top)"

            def panel_size():
                l, t, r, b = api.rect(me)
                return r - l, b - t

            def hide():
                self.root.withdraw()

            def show():
                self.root.deiconify()
                self.root.update_idletasks()

            self.docker = window_dock.Docker(api, sim, me, panel_size, show, hide)
            side = self.docker.start()
            try:
                self.root.attributes("-topmost", side == "overlap")   # beside it: no need
            except self.tk.TclError:
                pass
            return {"right": "docked to the right of the 3D window",
                    "left": "docked to the left of the 3D window",
                    "overlap": "placed over the 3D window's edge (no room beside it)"}[side]
        except Exception as exc:                     # never let docking break the mission
            self.docker = None
            return f"not docked ({type(exc).__name__}: {exc}); panel kept on top"

    # --------------------------------------------------------------------- API --
    def poll(self):
        """Process window events; return the operator actions since the last call.
        Actions: ("start",) ("select", zone) ("cmd", Cmd) ("heat",) ("cam",)."""
        if self.closed:
            return []
        if self.docker is not None:
            try:
                self.docker.poll()
            except Exception:                        # e.g. 3D window gone: stop docking
                self.docker = None
        try:
            self.root.update()
            if not self.root.winfo_exists():
                self.closed = True
        except self.tk.TclError:          # window was destroyed
            self.closed = True
        ev, self.events = self.events, []
        return ev

    def refresh(self, mission, selected, status_of, last_result=None, camera_mode=None,
                status_text=None):
        """Redraw everything that depends on mission state. status_of(zone) -> status key.
        last_result: the latest CommandResult (or None; its message is in mission.history).
        status_text: extra status line, e.g. navigation / detection state."""
        if self.closed:
            return
        try:
            self._refresh(mission, selected, status_of, last_result, camera_mode, status_text)
        except self.tk.TclError:          # window went away mid-update
            self.closed = True

    def _refresh(self, mission, selected, status_of, last_result, camera_mode, status_text=None):
        state = mission.state
        self._set(self.mission_lbl, text=f"Mission: {state}",
                  fg=OK_FG if state == "SEARCHING" else SUB_FG)
        if state == "IDLE":
            self._set(self.start_btn, text="START MISSION", relief="raised")
            self.start_btn._colours = ("#2e7d32", "#1b5e20")
        else:                                       # greyed out once running
            self._set(self.start_btn, text="Mission started", bg="#9e9e9e", relief="flat")
            self.start_btn._colours = ("#9e9e9e", "#9e9e9e")
        for z, lbl in self.zone_lbls.items():
            st = status_of(z)
            sel = z == selected
            if self.compact:        # 3 lines: selection shown by > < and the black outline
                text = f"{'> ' if sel else ''}ZONE {z}{' <' if sel else ''}\n{lbl._direction}\n{STATUS_TEXT[st]}"
            else:
                text = f"ZONE {z}\n{lbl._direction}\n{STATUS_TEXT[st]}\n "
                if sel:
                    text = f"> ZONE {z} <\n{lbl._direction}\n{STATUS_TEXT[st]}\nSELECTED"
            self._set(lbl, text=text, bg=ZONE_FILL[st], relief="sunken" if sel else "raised")
            self._set(self.zone_frames[z], bg="#111111" if sel else PANEL_BG)
        self._set(self.cmd_head, text=f"2. Command for Zone {selected}")
        for cmd, b in self.cmd_btns.items():
            self._set(b, text=cmd.value.replace("selected zone", f"Zone {selected}")
                      if cmd in self.zone_commands else cmd.value)
        self._collect_messages(mission)
        if self._latest is not None:                    # newest message, whatever its source
            msg, kind = self._latest
            self._set(self.result_lbl, text=msg,
                      fg={"ok": OK_FG, "bad": BAD_FG, "event": EVENT_FG}.get(kind, "#222222"),
                      bg=LATEST_BG[kind])
        if status_text is not None:
            self._set(self.status_lbl, text=status_text)
        if not self._resized:       # labels now hold real text: fit the window to it once
            self._resized = True
            self._size_to_screen()
        nxt = mission.plan[0] if mission.plan else "none"
        self._set(self.plan_lbl, text=(f"Next zone: {nxt}\n"
                                       f"Plan: {mission.plan_text()}\n"
                                       f"Searched: {', '.join(mission.visited) or 'none'}    "
                                       f"Excluded: {', '.join(mission.excluded) or 'none'}"))
        if camera_mode:
            self._set(self.cam_btn, text=f"Camera: {camera_mode}")

    def close(self):
        """Close from code (end of the simulation): no confirmation."""
        if not self.closed:
            self._on_close()


def try_open(*args, **kwargs):
    """OperatorPanel, or None (with the reason) if Tkinter / a display isn't available."""
    try:
        return OperatorPanel(*args, **kwargs), None
    except Exception as exc:                     # ImportError, TclError (no display), ...
        return None, f"{type(exc).__name__}: {exc}"
