# PBI-10: Maintain information and current mission state

If the program crashes mid-mission, the drone's mission state is already on disk and
`python main.py --resume` carries on from it.

## What counts as mission state
Saved in `mission_state.json` (next to `main.py`):

| Part | Contents |
|---|---|
| mission | state, plan (zones still to search), visited zones, destination, last response, full history log of operator commands and system responses |
| setup | the operator's starting information: last known position (lat/lon), search area, map size, OSM file, seed, visuals, camera mode, time limit |
| zones | zone centres A-D used in that run |
| drone | last known position `[x, y, z]` |

Not saved (restarts fresh after a resume): probability map and detection confidence.

## How to use
```bash
python main.py                 # new mission; an old state file is kept as mission_state.json.previous
python main.py --resume        # continue the last saved mission (skips the setup window, rebuilds the same world)
python main.py --state-file my_mission.json   # use a different file
python test_mission_state.py   # run the tests (no PyBullet needed)
python -m unittest -v test_mission_state_extra   # run the extra tests (no PyBullet needed)
python -m unittest test_mission_state test_mission_state_extra   # run all 68 tests together
```
Add to `.gitignore`: `mission_state.json*` (it is runtime data, not source).

### What the operator sees on start-up
* `--resume` with a good file: prints on a one-line summary of the saved mission and which file it came from (main or backup), then skips the setup window.
* `--resume` with a missing or unusable file: prints why each file was skipped and starts a new mission.
* No `--resume` but an unfinished mission (state SEARCHING) is on disk: prints a reminder that --resume exists, then starts a new mission and keeps the old file as .previous.
* On resume, the drone returns to its saved position only if that position is inside the map and at a sensible altitude (0-20 m). Otherwise it starts from the launch point.
* On resume while SEARCHING, the drone is sent back to its saved destination. If the saved zones no longer match, a note is printed and logged.
* Every resume adds a "restore" entry to the history log, so repeated resumes are visible.

## When it saves
* Immediately on any mission change (state, plan, visited, destination, response, new history line).
* A new history line is still noticed and saved when the history is already at its 2000-entry cap (older lines are dropped, so the length no longer changes).
* While SEARCHING, the drone position is refreshed every 0.5 s.
* Drone movement alone is not saved when the mission is not SEARCHING. Periodic refreshes skip the fsync, while mission changes are always fsynced.
* Once at start-up and once more on a normal exit or Ctrl+C.
* A save failure (full disk, read-only folder) prints one warning and the mission continues.

## Storage option chosen (research summary)
| Option | Verdict |
|---|---|
| **JSON file, atomic write** | **Chosen.** Plain text, no dependencies, nothing executes when loaded, easy for the team to inspect. |
| pickle | Rejected: loading a tampered file can run arbitrary code. |
| SQLite | Safe and crash-proof, but more machinery than a single small state record needs. Good option if the team later wants queryable history. |

Safety measures in `mission_state.py`: write to a temp file in the same folder, flush + fsync, then
`os.replace()` (atomic swap); keep the previous good copy as `.bak`; SHA-256 checksum; strict
validation of every field on load; size cap; load never raises (bad file -> backup -> fresh start).
### Additional details of the safety measures: 
* The checksum covers the values, not the key order, so harmless reformatting does not invalidate a file.
* The folder is fsynced after the swap, so the rename itself survives a power cut.
* A crash between the two renames still recovers from the .bak copy. A failed write leaves the old file intact and no temp files behind.
* A good main file is used even if the backup is corrupt. A missing main file with a good backup loads the backup. If both are bad, each file is reported by name.
* `load()` never modifies or deletes files.
* Files with the wrong layout, non-ASCII bytes, a wrong version, non-finite drone positions, or unhashable zone names are rejected, not crashed on.

## Changes to existing code
* `operator_guidance.py`: `Mission.snapshot()/restore()/validate_snapshot()`, `record()` and `finish()`
  for logged system responses, `arrived()` now logs, and two fixes (below).
* `main.py`: `--resume` / `--state-file`, restore on start-up, autosave once per render tick, and the
  ABORT / COMPLETE transitions now go through `mission.finish()` so they appear in the history.
* `mission_state.py`: new module with `MissionStateStore (build, save, autosave, load, archive_previous)`, `validate_setup, validate_payload, setup_from_args, apply_setup_to_args` and describe.
* `test_mission_state_extra.py`: new test file (39 tests).

## Bugs found and fixed
1. `Mission.handle()` crashed with `AttributeError` for any non-preset input (for example `None` or a
   string) when the mission had not started or had ended. It now rejects it with a message.
2. After a mission ended, `START` said "mission already started", which was misleading. It now says
   the mission is FOUND / ABORT / COMPLETE.
3. System responses set directly in `main.py` (abort, plan complete, returned to launch) and zone
   arrivals were not in the history log. They are now.
4. Autosave did not notice a new history line once the history was at its cap. It does now.
5. `load()` raised on a signed file whose zone names were unhashable. It now rejects the file safely and falls back to the backup.

## Test results (`python test_mission_state.py`)
68 tests, all passing (29 in `test_mission_state.py` + 39 in `test_mission_state_extra.py`).

| Acceptance criterion | Evidence |
|---|---|
| Saved within 1 s of any change | Every kind of change saved on the same tick; 68 timed saves: mean 3.9 ms, max 4.6 ms (budget 1000 ms), including a history at its 2000-entry cap |
| Restart loads the last saved state | 16 crash scenarios in real subprocesses (crash after each of 8 steps, with the last change both saved and unsaved), plus 20 random SIGKILLs: the file always loaded and never lost an acknowledged save |
| Commands before start / after end rejected, no crash | All presets rejected before START and after FOUND / ABORT / COMPLETE; 12 kinds of odd input never raise |
| Every command and response in the history | One history line per command, rejection, arrival and system response; history survives save and restore |

Also tested: truncated, edited, wrong-version, empty, oversized and randomly corrupted files (300 mutations).
Also tested in the extra file: setup validation (wrong types, lonely coordinate, defaults), `describe()` summaries, `--state-file` handling, and archiving the old file on a fresh run.

## Not tested here
`main.py` itself (the `--resume` wiring and the per-tick autosave) was not run, because PyBullet could
not be installed in the environment used to write this. Please run it once on a real machine:
start a mission, close the window mid-search, then run `python main.py --resume`.
