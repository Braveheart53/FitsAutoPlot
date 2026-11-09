# FitsAutoPlot

**Python 3.8 and 3.12** scripts (tested 2026-11-09, see REQUIREMENTS.md) for autoplotting FITS files using Veusz and astropy. This allows any user to view the final
data sets and alter the plots directly using vuesz without knowing any python. Includes other post processing routines
as well for various data strucure parsing such as Keysight, Agilent, OrbitFR, and Rhode and Schwarz Test Equipment

# Files these scripts can currently reduce

|               File Description               | Extennsion 1 | Extension 2 |  Extension 3  |                                                                                                                                                    Status                                                                                                                                                    |
|:--------------------------------------------:|:------------:|:-----------:|:-------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                  FITS files                  |     .fit     |    .fits    |   .fits.gz    | FITS_AutoPlot 0.4.0: astropy reader, native .vszh5 output (no Veusz needed to write), interactive pyqtgraph Plots tab. Every page draws DMJD as a Veusz date-time axis in local ISO 8601 (2026-11-09T08:39:00-05:00); small gaps break the line, large gaps break the x axis (both set in the GUI); padding rows (DMJD = 0) are left out; the full FITS header is in each page's notes. Checked with GBT sampler2log files (azServo1Hz, Rcvr_2500 Temperature/Vacuum, SiteTime OnePpsDeltas). |
| Rhode & Schwarz spectrum analyzer trace files |     .sft     |             |               | RAndS_FSW_ASCII_Plotter 1.1.2: FSP/FSW "Section" files (ASCII and binary) and FSW x;y exports; frequency or zero-span time axis from the header; power averages per file and for all files; pyqtgraph Plots tab; native .vszh5. |
|            Comma Separated Files             |     .csv     |    .tsv     |  .txt / .dat  | CSV_TSV_AutoPlot 6.1.2: any delimiter, numeric and date/time columns (Veusz date-times shown in local ISO 8601), rows kept aligned, pyqtgraph Plots tab, native .vszh5. |
|         GBO Outdoor Range Data Files         |     .atr     |             |               | ATR_AutoPlot 0.2.2: Cartesian and polar magnitude / phase pages, multi-frequency files, overlays; pyqtgraph Plots tab with polar plots; native .vszh5. |
|               Touchstone Files               |     .s1p     |    .s2p     | .s\*p or .snp | Touchstone_AutoPlot 0.7.1 (Smith chart pages in the Veusz project, drawn with Veusz widgets by vszh5_writer 0.0.26; pyqtgraph previews and Plots tab; see RELEASE_suite-2026-11-09.md); 0.5.0: magnitude / phase (wrapped or unwrapped), scikit-rf time domain and time gating (auto or manual, band-pass or band-stop) shown in time and frequency, Smith charts (PNG/SVG/PDF/HTML, bookmarked combined PDF), native .vszh5 output without Veusz. Checked on the 223 example .s2p files (220 processed; 3 are malformed and are reported as failures); the projects load and render in Veusz 4.2.1. |
|        Franks timing data (FranksProcessed)  |   .t00new    | time_gbt.dat |               | Franks_AutoPlot 0.4.0: native .vszh5 output, the same pyqtgraph Plots tab, local ISO 8601 time axes and small/large gap controls as FITS AutoPlot. |
|        ORBIT/FR (MVG 959 Spectrum) data      |              |             |               | Not written yet: the vendor COM API is not public. See ORBIT_FR/ORBIT_FR_API_REQUIREMENTS.md and run ORBIT_FR/orbitfr_com_probe.py on the 959 Spectrum PC. |

# postProcessing_Launcher.py

GUI for launching the various scripts / tools utilized.
This should be your starting point if you are just running the scripts for plotting data and not developing.

Version 0.2.1 (2026-11-09): real icons on every button (assets/, drawn by tools/make_suite_icons.py; the app icon is from GBT_2.ico). Version 0.2.0 (2026-11-05): one button and one Applications-menu entry (Ctrl+1 ...) per tool, including Franks_AutoPlot; View > Light / Dark mode; Help > About (author, NRAO/GBO, licence) and Veusz plugin instructions; per-launch log files (File > Open Launch Log Folder) with a report if a tool exits with an error at start-up. Run `python postProcessing_Launcher.py --list` to see which tools it found. A frozen (PyInstaller) launcher needs `GBO_POSTPROC_PYTHON` set to the python.exe of the plotting environment. Details: RELEASE_launcher-0.2.0_touchstone-0.5.0.md.

# Common to every tool (2026-11-09)

* **Project files**: every `.vszh5` is written by the shared writer `vszh5_writer.py` 0.0.27 (through
  `_autoplot_native.py` 0.5.0). Veusz is not needed to write a project, only to open it, or for
  File > Export PDF / PNG.
* **Plots tab**: every tool shows the pages of its project with pyqtgraph (`_autoplot_pyqtgraph.py`):
  values at the cursor, Ctrl+wheel zoom, CSV/PNG/SVG export, polar (ATR) and Smith (Touchstone) plots.
* **Light / dark**: View > Light Mode / Dark Mode is one setting for the launcher and every tool
  (`_suite_gui.py`); the Veusz plugins style only their own dialog.
* **Date-times**: time stamps are stored as Veusz date-time datasets in UTC plus a `<name>_local`
  copy on the clock of the computer doing the processing (daylight saving included). Date-time
  axes read `2026-11-09T08:39:00-05:00` (ISO 8601, 24 h, with the UTC offset); when the data cross
  a daylight-saving change the ticks omit the offset and the axis label lists each offset and when it
  starts. Set `VSZH5_DISPLAY_TZ` (for example `America/New_York`, `-05:00` or `UTC`) to process for
  another zone, e.g. on a build server that runs in UTC.
* **Names and notes**: page and graph names follow the original scripts; traces are named after their
  y dataset; the source-file header is in each page's notes (Veusz: Properties > notes).
* **Gaps** (FITS / Franks, 0.4.0): a *small gap* - a step longer than K x the median step of that trace
  (default K = 10), or a fixed time - breaks the line; a *large gap* - longer than n x the small gap
  (default n = 100), or a fixed time such as `3d` - also breaks the x axis. n = 0 keeps one continuous
  axis. Gap texts take `s`, `min`, `h`, `d` and `m` (months).

Release notes: RELEASE_suite-2026-11-09.md (sections 1-9) and its FITS addendum (section 10).
Every option has a tooltip (hover over it for help).

# FITS-AutoPlot.py

Import FITS tables and images (astropy) and build a native Veusz project; see the table above.

# GBO_antenna_postProcessing.py

The intent of this script is to be a combined data reduction tool with a Qt GUI.
The current embodiment (branch 0.0.1) is far from the intent but does include the atr data reduction functions.
In the end, it should import various scripts for each file type rather than have all classes and functions within
itself.
Therefore, the atr reduction should come out to its own script in future work.
This has been replaced by postProcessing_Launcer for the time being.

# RAndS_FSW_ASCII_Plotter.py

Imports Rohde & Schwarz spectrum-analyzer trace files (.sft): every sweep, the power average of each file
(mean of mW, not of dB) and of all files sharing one x axis. Zero-span files (x-Unit s) get a time axis.
Batch use: `python RAndS_FSW_ASCII_Plotter.py --mode 0 --output out.vszh5 *.sft`.

# fastest_ascii_import.py

A single function that parses any ASCII file for given line numbers and/or search strings and returns a dict
with all extracted information (0.0.3: strict data-line detection, any byte decodes).

# Revisions

Symantic Revisioning with the following
structure: [external or production release].[internal release for review].[draft or work in progress relases / updates]

## Revision By number

Many updates are applied and well documented through GIT commits. If you have any questions, please contact the
developer or submit an issue request.

# Required Python packages

The suite runs on **Python 3.8** and **Python 3.12**; both were tested on 2026-11-09 with the
versions in REQUIREMENTS.md (required and optional packages, what each is for, 3.8 vs 3.12
differences, test results). Create an environment with mamba:

````bash
mamba env create -f environment-py312.yml     # or environment-py38.yml
mamba activate fitsautoplot-py312
python postProcessing_Launcher.py
````

`requirements-py38.txt` / `requirements-py312.txt` list the exact tested wheels (pip/uv). Veusz
(viewer), CuPy (GPU) and pywin32 (ORBIT/FR probe) are optional.

A [detailed google doc is available for Python Environment Creation Direction](<https://docs.google.com/document/d/1roQzx02ZDnD8I1MvyUEJWbL53ArOPD9zG1YbmjY1jFs/view?usp=sharing>)
