# PBI-07: Operator dashboard responsiveness under heavy input

**Target:** an operator guidance click is processed and shown within **50 ms**. The dashboard and the drone display keep updating within 50 ms while heavy live data comes in, and nothing crashes.

## 1. What was slow (analysis of the original code)

All numbers were measured on the development laptop (Windows 11). The original code is commit `496939c` (`main`).

| # | Where | Problem | Measured |
|---|-------|---------|----------|
| 1 | `main.py` panel poll | Clicks were only read at 20 Hz. | A click waited up to 50 ms before it was seen. |
| 2 | `apply_operator_command` | The probability-map update and the A* route re-plan ran **before** the panel was updated. | Re-plan took 5–27 ms in the test world, and longer in larger OSM worlds. |
| 3 | `operator_panel.refresh` | The reply was only drawn at the next poll. | Up to 50 ms more. |
| 4 | Panel message log | Live messages were written to Tk one line at a time. | 1,000 messages took 440 ms; 2,000 took 810 ms. |
| 5 | Panel layout | When a reply or event wrapped to more or fewer lines, Tk laid out the whole panel again. | 40–70 ms per change. |
| 6 | `MissionStateStore.autosave` | Every mission change was saved on the UI thread. The payload was serialised twice, then `fsync`ed. | 6–34 ms per frame while data streams in. |
| 7 | Heatmap (3D view) | Each tile recolour waits for PyBullet's render thread. There were 8 tiles per frame, and a show/hide toggle recoloured all ~900 tiles. | 30–130 ms per frame; about 3 s frozen per toggle. |
| 8 | HUD and zone markers (3D view) | After a command, about 25 debug text and line calls each waited for the render thread. | 80–120 ms. |

## 2. What changed

| File | Change |
|------|--------|
| `ui_responsiveness.py` (new) | **`CommandPipeline`** handles a click in two stages. Stage 1 validates the command and updates the mission (< 0.1 ms), so the panel can show the reply at once. Stage 2 runs the probability map update and the route re-plan after the reply is on screen. Several clicks before stage 2 runs share **one** re-plan to the newest destination; the end state is the same (tested). **`ResponseTimer`** records click → drawn times and checks them against 50 ms. |
| `main.py` | Clicks are read every frame (60 Hz). Order per click: stage 1 → `panel.refresh` → `panel.flush` (drawn) → stage 2. The panel status is refreshed and drawn every 2 frames (33 ms); the HUD also refreshes every 2 frames. The heatmap is repainted in one batch with rendering paused, 4 times a second; the toggle only flips a flag. HUD, zone marker and route drawing also run with rendering paused (`rendering_paused`). Saves go through `BackgroundSaver`. The end-of-run summary prints `UI response - ...`. |
| `operator_panel.py` | The log is written in one Tk insert per refresh, keeping at most `LOG_KEEP` lines. Boxes whose text changes live have a fixed number of lines, so new text never forces a relayout. New `flush()` draws immediately. `poll()` records when a click came in (`input_t`). Button-flash timers are cancelled on close; this fixes the `invalid command name` errors that appeared when the panel closed within 350 ms of a click. |
| `mission_state.py` | **`BackgroundSaver`**: the UI thread only copies the state (about 0.3 ms). A worker thread writes the JSON, the checksum and the file, then `fsync`s it. The newest snapshot wins, and saves are at most 0.1 s apart. `close()` writes the final state and waits for it. The payload is serialised once and reused for the checksum and the file. Old files still load. |
| `probability_map.py` | `draw_heatmap(budget=None, pause_rendering=True)` recolours every changed tile in one batch. Hiding the heatmap is handled the same way (tiles are made transparent). |

**Why the re-plan is not on a thread.** The re-plan casts rays and draws through PyBullet, and it changes the navigator that the flight loop is using. Neither PyBullet nor Tk may be used from two threads. So it runs **after** the UI is drawn instead. Disk saves only need a copy of the data, so they do run on a worker thread.

## 3. Test cases

Run from `Sprint3/`:

```
python -m unittest tests.test_ui_responsiveness_pbi07 -v
```

The tests use the real Tk panel (skipped if there is no display), the real `Mission`, and a 200 × 200 probability map. The route re-plan is replaced by a 40 ms CPU-bound stand-in, which is slower than the real one. Clicks are real Tk events sent to the panel's buttons.

| AC | Test | Result (3 runs) |
|----|------|-----------------|
| AC-02 | 80 clicks (zone selects and commands) with a 40 ms re-plan: click → drawn | worst 9.9–11.5 ms |
| AC-02 | Same click, old order vs new order | about 70 ms vs 0.5–1.5 ms |
| AC-02 | Worst case including the wait for the next poll (16.7 ms) | 24–25 ms |
| AC-02 | The reply is drawn **before** the re-plan runs | pass |
| AC-02 | A burst of 11 commands gives the same mission, map and history as one at a time, with **1** re-plan | pass |
| AC-01 | Bursts of 10 to 5,000 new messages: refresh + draw | 10–19 ms; the log stays at 300 lines |
| AC-01 | Bursts of 50 to 1,000 events | 25–27 ms |
| AC-01 | 10 s stream: 1,200 messages/s, an event every frame, map updates, 2 clicks/s | worst frame 10–11 ms; worst click 8–9 ms; 0 over 50 ms |
| AC-01 | Odd text (empty, 5,000 characters, emoji, Tk special characters) | no crash |
| AC-01 | Autosave cost on the UI thread (2,000-line history, 200 × 200 map) | inline 21.8 ms mean / 36 ms worst → background 0.66 ms / 2.7 ms |
| AC-01 | Background saver: newest state on disk after close and loads cleanly; the copy is unaffected by later changes; a disk error doesn't crash or block | pass |
| AC-03 | 4 s real-time loop with heavy input: drone-display change → drawn | worst 31–37 ms; 0 over 50 ms |
| AC-03 | HUD, panel and poll cadence | 33 / 33 / 17 ms |

**Full suite:** `python -m unittest discover -s tests`: **363 tests OK.** Three earlier tests were adjusted. Two now patch the panel's batch log method instead of the per-line one, and the PBI-08 cadence test reads `PANEL_REFRESH_EVERY`.

## 4. Full simulation (real PyBullet GUI + operator panel)

A driver script sent START, 15 guidance clicks (including a burst of 3 within 0.1 s), two heatmap toggles and 30 extra live log lines per mission message into the real panel. It ran `main.py --no-ui --simple-visuals --max-time 40 --profile` once on the original code and once on the PBI-07 code, with the same script.

| Measure (main.py's own end-of-run report) | Original | PBI-07 |
|---|---|---|
| Panel update time: mean / p95 / worst | 372 / 1,315 / **8,502 ms** | 13.5 / 31 / **38 ms** |
| HUD (3D) update time: mean / p95 / worst | 98 / 178 / **3,118 ms** | 15 / 34 / **40 ms** |
| Click → reply drawn on the panel | not measured (≥ 50 ms by design) | **worst 11.3 ms** (19 clicks) |
| Command confirmation on the HUD | 30–117 ms* | 6–22 ms |
| Heatmap toggle | about 3 s frozen | 2–9 ms |
| Heatmap cost per frame while searching | 33–37 ms mean | 3–4 ms mean |
| Re-plans skipped (burst of clicks) | – | 2 |
| Saved state after exit | loads OK | loads OK |

\* The original measured from when processing started, not from the click.

## 5. Known limits (minor, for future work)

- **Busy stretches of the flight loop:** a frame can still take up to about 50 ms when the navigator re-plans around an obstacle during flight. This is the `logic` section of `--profile`, not operator input. The drone display stayed within 40 ms in the GUI run.
- **Panel text boxes:** the latest message box shows 3 lines (2 on short screens), and longer text is cut at the bottom of the box. The full message is always in the log below it.
- **Saves:** with the GUI, saves are written up to about 0.1 s after a change. That is still well inside PBI-10's 1 s limit. A normal exit, Ctrl+C or an error always writes the final state.
- **Hardware:** the timing figures come from one Windows laptop. Very slow machines may need the GUI run repeated.
