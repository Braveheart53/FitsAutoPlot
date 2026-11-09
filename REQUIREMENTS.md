# FitsAutoPlot suite — requirements and tested environments

Revision 0.1.0 (2026-11-09). Applies to the 2026-11-09 suite (FITS / Franks AutoPlot 0.4.0,
writer 0.0.27). Author: William W. Wallace (wwallace@nrao.edu).

The suite runs on **Python 3.8** (the RHEL 8 / NRAO set-up) and **Python 3.12**. Both were tested
on 2026-11-09 with the exact package versions below: the unit tests, every program run without a
display, and the four sets of GBT FITS files. The two versions wrote the same projects: the same
dataset names, identical arrays, and the same Veusz script.

## 1. Install (mamba)

```bash
# Miniforge (includes mamba): https://github.com/conda-forge/miniforge
mamba env create -f environment-py312.yml        # or environment-py38.yml
mamba activate fitsautoplot-py312                # or fitsautoplot-py38
python postProcessing_Launcher.py                # the launcher; each button starts one program
```

Each `environment-py*.yml` file gets the interpreter from conda-forge. pip then installs the
tested wheels listed in `requirements-py*.txt`, so the new environment matches the tested one.
Without mamba, use `python3.12 -m pip install -r requirements-py312.txt` (or the 3.8 file).
On an NRAO RHEL 8 host, `setup_py3p8_nrao.sh` still builds the `Py3p8` environment.

Update an environment after a new release:
`mamba env update -f environment-py312.yml --prune`.

## 2. Packages

### 2.1 Required (both Pythons)

| Package | Python 3.8 (tested) | Python 3.12 (tested) | Used by | Why |
|---|---|---|---|---|
| numpy | 1.24.4 | 2.5.4 | all | arrays |
| h5py | 3.11.0 | 3.16.0 | all (vszh5_writer) | writes the `.vszh5` HDF5 projects; Veusz is not needed to write them |
| astropy (+ pyerfa) | 5.2.2 (2.0.0.3) | 8.0.2 (2.0.1.5) | FITS AutoPlot and plugin | FITS reader, MJD conversion |
| psutil | 7.2.2 | 7.2.2 | FITS / Franks | RSS memory monitor; arrays spill to disk above the threshold |
| qtpy | 2.4.3 | 2.4.3 | every GUI | Qt abstraction layer |
| PySide6 | 6.6.3.1 | 6.12.0 | every GUI | Qt 6 binding used through qtpy |
| pyqtgraph | 0.13.3 | 0.14.0 | every GUI (Plots tab) | interactive plots; 0.13.3 is the last release for 3.8 |
| scipy | 1.10.1 | 1.18.1 | Touchstone, CSV/TSV | scikit-rf; statistics |
| pandas | 2.0.3 | 3.1.1 | CSV/TSV AutoPlot | delimited-file reader |
| scikit-rf | 1.9.0 | 2.2.0 | Touchstone AutoPlot | Touchstone reader, time domain, gating |
| matplotlib | 3.7.5 | 3.11.2 | Touchstone; tools/make_suite_icons.py | PDF/PNG Smith-chart export; icons |
| mpld3 | 0.5.12 | 0.5.12 | Touchstone | HTML Smith charts |
| pypdf | 5.9.0 | 7.0.0 | vszh5_writer, Touchstone | PDF bookmarks (outline) |
| pillow | 10.4.0 | 13.0.0 | matplotlib; icons | images |
| tzdata | 2026.6 | 2026.6 | vszh5_writer | IANA time-zone database. Windows has none, and `zoneinfo` needs this package there |
| python-dateutil | 3.0.0 | 4.0.0 | vszh5_writer on 3.8 | IANA zone names on 3.8, which has no `zoneinfo`. Optional on 3.12 |

### 2.2 Optional

| Package | Needed for | Without it |
|---|---|---|
| Veusz (3.6.2 or 4.2.1 tested) | opening and rendering the `.vszh5` projects, *Open in Veusz*, *Export PDF/PNG*, the two Veusz plugins | everything else works; the projects are still written |
| CuPy (`cupy-cuda12x`) + NVIDIA GPU | GPU sorting (FITS / Franks) and GPU averaging (R&S) | NumPy on the CPU (same results) |
| backports.zoneinfo | IANA zone names on Python 3.8 (alternative to python-dateutil) | python-dateutil is used |
| pywin32 (Windows) | `ORBIT_FR/orbitfr_com_probe.py` | the probe cannot run |
| PyInstaller / pyside6-deploy | frozen launcher (`postProcessing_Launcher.spec`, `pysidedeploy.spec`) | run from source |

### 2.3 Python 3.8 versus 3.12

* **Source.** The code is written to Python 3.8 grammar. A unit test parses every `.py` file with
  `feature_version=(3, 8)`. `from __future__ import annotations` keeps 3.9+ annotations such as
  `list[str]` from being evaluated.
* **Time zones.** `zoneinfo` arrived in Python 3.9. On 3.8, `vszh5_writer.iana_zone()` tries
  `backports.zoneinfo` and then `dateutil.tz`. Writer 0.0.26 failed on 3.8 with
  `ModuleNotFoundError: zoneinfo` when `VSZH5_DISPLAY_TZ=America/New_York` was set; 0.0.27 fixes this.
  An unknown zone name now gives a warning, and the computer's own zone is used.
* **Library ranges.** pyqtgraph 0.13.3, scikit-rf 1.9.0, astropy 5.2.2 and PySide6 6.6.3.1 are the
  newest releases that install on 3.8. The code handles API differences between these versions,
  for example skrf `time_gate` parameters and pyqtgraph enums.
* **Multiprocessing.** Every pool uses the `spawn` start method on all platforms. Python 3.8 keeps a
  relative path for the main script, so start programs from their own folder or by full path. The
  launcher passes full paths.

## 3. Tested on 2026-11-09

Host: Linux x86_64 with Qt `offscreen`. `VSZH5_DISPLAY_TZ=America/New_York`, so the local axes
show EST/EDT as at Green Bank.

| Check | Python 3.8.20 | Python 3.12.3 | Python 3.13 (+ Veusz 4.2.1) |
|---|---|---|---|
| `python -m unittest discover -s tests` (41 tests) | OK (every repeat run since the fix) | OK (every repeat run since the fix) | OK |
| Close-window stress (`tests/test_gui_close.py`): FITS/Franks windows with the memory monitor firing from its thread and a populated Plots tab, plugin dialogs, ATR/CSV/R&S/Touchstone windows; open/close/delete twice each while `gc.collect()` runs in a second thread every 2 ms | OK | OK | OK |
| FITS / Franks date-time pages (`dt_pages`) | pass | pass | pass |
| FITS: tack (10 files), UWBR Temps (15), UWBR Vacuum (15), 30d OnePpsDeltas (319) | pass | pass | pass + rendered in Veusz |
| ATR, CSV/TSV, R&S GUIs (headless: open, process, Plots tab, save) | pass | pass | pass |
| Touchstone: 223 example `.s2p` files, pyqtgraph previews, Smith pages | pass | pass | pass |
| Plots tab (pyqtgraph) light / dark | pass | pass | pass |
| Launcher: splash, buttons with icons, menus, About, launch/reap | pass | pass | pass |
| Same projects from 3.8 and 3.12 (tack 776, Temps 384, Vacuum 95, 30d 40 195 datasets) | identical | identical | — |

**Found and fixed in testing.** Under Python 3.12 / PySide6 6.12, the unit tests crashed with a
segmentation fault in 3 of 4 runs. The cause was in the program, not the tests. The memory-monitor
thread held a lambda that referred to the window or plugin dialog. That thread could release the
last reference and destroy Qt widgets outside the GUI thread. The monitor also wrote to the log
widget from its own thread. In `_autoplot_common` 0.4.0 and the plugins 0.3.0, the callback is
held weakly and logging is queued to the GUI thread. Since the fix, no run on 3.8, 3.12 or 3.13
has failed (more than 20 full runs). `tests/test_gui_close.py` checks the fix directly: it
closes each window while the monitor and a garbage-collector thread are running.

## 4. Environment variables

| Variable | Meaning |
|---|---|
| `VSZH5_DISPLAY_TZ` | Zone for the local date-time axes: `America/New_York`, `-05:00`, `UTC-05:00` or `UTC`. Unset means the computer's zone |
| `GBO_POSTPROC_PYTHON` | python.exe that a frozen launcher uses to start the programs |
| `QT_QPA_PLATFORM=offscreen` | Run the GUIs without a display (tests) |
