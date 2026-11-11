# FitsAutoPlot suite — requirements and tested environments

Revision 0.2.1 (2026-11-10). Applies to the 2026-11-10 suite (FITS / Franks AutoPlot 0.4.2,
R&S 1.1.4, ATR 0.2.4, CSV/TSV 6.1.3, Touchstone 0.7.3, writer 0.0.29). 0.2.1: mpld3 0.5.10 on
Python 3.8 (0.5.11 and 0.5.12 call `Axis.get_converter`, which matplotlib 3.7.5 lacks), and the pandas parser note below. 0.2.0 added
the GPU options for NVIDIA and AMD (section 2.4) and the process-pool rule (section 2.3).

pandas and exact numbers (0.2.1): the default C float parser of pandas 2.0.3 (the newest for
Python 3.8) and 3.0.5 is not exact for numbers with more than 15 significant digits or a power of
ten beyond +/-22 (measured: up to 7 375 ulp, relative 1e-12); pandas 3.1.1 is exact. CSV/TSV
AutoPlot 6.1.3 checks the installed pandas once and reads such files with
`float_precision="round_trip"`, so any of these versions gives the same, exact values. Author: William W. Wallace (wwallace@nrao.edu).

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
| mpld3 | 0.5.10 | 0.5.12 | Touchstone | HTML Smith charts. On 3.8 use 0.5.10: 0.5.11 and 0.5.12 call `Axis.get_converter`, which matplotlib 3.7.5 (the newest for 3.8) does not have (tested 2026-11-10) |
| pypdf | 5.9.0 | 7.0.0 | vszh5_writer, Touchstone | PDF bookmarks (outline) |
| pillow | 10.4.0 | 13.0.0 | matplotlib; icons | images |
| tzdata | 2026.6 | 2026.6 | vszh5_writer | IANA time-zone database. Windows has none, and `zoneinfo` needs this package there |
| python-dateutil | 3.0.0 | 4.0.0 | vszh5_writer on 3.8 | IANA zone names on 3.8, which has no `zoneinfo`. Optional on 3.12 |

### 2.2 Optional

| Package | Needed for | Without it |
|---|---|---|
| Veusz (3.6.2 or 4.2.1 tested) | opening and rendering the `.vszh5` projects, *Open in Veusz*, *Export PDF/PNG*, the two Veusz plugins | everything else works; the projects are still written |
| CuPy (NVIDIA: `cupy-cuda12x` / `cupy-cuda13x`; AMD: `cupy-rocm-7-0`) or PyOpenCL (any OpenCL GPU) | GPU sorting (FITS / Franks), GPU dB averaging (R&S), GPU time-domain transform (Touchstone, with pyvkfft on OpenCL) - only for large arrays; see section 2.4 | NumPy on the CPU (same results) |
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
* **When a process pool is used** (`_autoplot_parallel.py`, 2026-11-09). FITS, Franks, ATR, R&S and
  Touchstone time the first two files in the program's own process and start a pool only when
  t_seq - (t_start + 1.1 t_seq / W) >= 0.5 s, where t_seq is the estimated time for the rest in one
  process, W the number of workers and t_start the pool start-up: 0.3 s (Linux) or 1.0 s (Windows)
  plus the program's import time, which every `spawn` worker pays again. The log says which was
  chosen. A main script without an `if __name__ == "__main__":` guard never gets a pool (its workers
  would run it again). The Veusz plugins use threads only.

### 2.4 GPU: NVIDIA and AMD (optional)

`_autoplot_gpu.py` 0.0.1 picks the first back end that finds a device; `AUTOPLOT_GPU` selects one
(section 4). Install one package into the environment (mamba or pip):

| GPU | Platform | Package (Python 3.10 - 3.13) | Python 3.8 | Checked here |
|---|---|---|---|---|
| NVIDIA | Windows, Linux | `cupy-cuda12x` or `cupy-cuda13x` 14.2.0 (CUDA 12 / 13 driver) | `cupy-cuda12x` 12.x (CuPy 13 dropped 3.8) | code path only (stand-in module; no NVIDIA GPU here) |
| AMD | Linux (ROCm 7) | `cupy-rocm-7-0` 14.2.0, or AMD's `amd-cupy` 13.5.1 (`--extra-index-url https://pypi.amd.com/rocm-7.0.2/simple`) | not available: use PyOpenCL | code path only (stand-in module) |
| AMD, NVIDIA, Intel | Windows, Linux | `pyopencl` 2026.1.4 + the GPU driver's OpenCL (ICD) | `pyopencl` 2024.1 | yes: PoCL 3.0 on the CPU, Python 3.8 (2024.1), 3.12 and 3.13 (2026.1.4) |
| (FFT on OpenCL) | Windows, Linux | `pyvkfft` (conda-forge has binaries; PyPI only source) | - | not installed (no binary here); the start-up self-test decides |

What runs on the GPU, and from which size (elements; `_autoplot_gpu.THRESHOLDS`, estimated -
there was no GPU to measure the break-even point):

| Operation | Used by | GPU from | Result versus NumPy |
|---|---|---|---|
| stable argsort (NaN last) | FITS / Franks (and plugins): time sort | 200 000 values | identical order (64-bit radix keys) |
| linear-domain dB mean `power_mean_db` | R&S: sweep and all-file averages | 1 000 000 values | within 1.4e-14 dB (OpenCL test) |
| centred inverse FFT | Touchstone: time domain | 1 000 000 values | within 1e-12 relative (self-test) |

Measured sizes in real use (2026-11-09): the largest FITS sort of the 319-file OnePpsDeltas batch had
28 212 values (all 13 119 sorts took 0.04 s), the Touchstone transforms of the 223 example files take
0.11 s together, and the 58 readable example R&S files have 1 to 299 sweeps of 501 to 801 points (at
most 149 799 values per average). So these files stay on the
CPU, which is faster for them; the GPU only helps much larger data. Each operation runs a small
self-test against NumPy on first use; a mismatch or error switches it to NumPy for good, and the tooltip
and log say so. The GPU libraries are imported on first use, not at start-up.

## 3. Tested on 2026-11-09

Host: Linux x86_64 with Qt `offscreen`. `VSZH5_DISPLAY_TZ=America/New_York`, so the local axes
show EST/EDT as at Green Bank.

| Check | Python 3.8.20 | Python 3.12.3 | Python 3.13 (+ Veusz 4.2.1) |
|---|---|---|---|
| `python -m unittest discover -s tests` (60 tests; 41 before 0.2.0 of this file) | OK (2 OpenCL tests skipped: PyOpenCL not in the tested set) | OK (same) | OK (same) |
| `tests/test_parallel_gpu.py` with PyOpenCL + PoCL 3.0 (OpenCL on the CPU) | OK (PyOpenCL 2024.1) | OK (2026.1.4) | OK (2026.1.4) |
| FITS tack batch with every sort on OpenCL (PoCL) | — | — | project identical (42 GPU sorts) |
| Close-window stress (`tests/test_gui_close.py`): FITS/Franks windows with the memory monitor firing from its thread and a populated Plots tab, plugin dialogs, ATR/CSV/R&S/Touchstone windows; open/close/delete twice each while `gc.collect()` runs in a second thread every 2 ms | OK | OK | OK |
| FITS / Franks date-time pages (`dt_pages`) | pass | pass | pass |
| FITS: tack (10 files), UWBR Temps (15), UWBR Vacuum (15), 30d OnePpsDeltas (319) | pass | pass | pass + rendered in Veusz |
| ATR, CSV/TSV, R&S GUIs (headless: open, process, Plots tab, save) | pass | pass | pass |
| Touchstone: 223 example `.s2p` files, pyqtgraph previews, Smith pages | pass | pass | pass |
| Plots tab (pyqtgraph) light / dark | pass | pass | pass |
| Launcher: splash, buttons with icons, menus, About, launch/reap | pass | pass | pass |
| Same projects from 3.8 and 3.12 (tack 776, Temps 384, Vacuum 95, 30d 40 195 datasets) | identical | identical | — |
| Same projects as the suite of the afternoon (writer 0.0.27): tack, Temps, 30d (3.8, 3.12, 3.13); R&S, Touchstone time domain (3.13) | identical | identical | identical (only the script's version comment differs) |

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
| `AUTOPLOT_GPU` | GPU back end: `auto` (default: CuPy, then an OpenCL GPU), `cuda`, `rocm`, `opencl` (also accepts an OpenCL CPU device, for tests) or `off` |
| `PYOPENCL_CTX` | Which OpenCL platform/device PyOpenCL uses (e.g. `0:1`), when there are several |
| `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` | Which NVIDIA / AMD GPU CuPy uses |
