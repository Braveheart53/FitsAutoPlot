# FitsAutoPlot

**Python 3.8** based scripts for autoplotting FITS files using Veusz and astropy. This allows any user to view the final
data sets and alter the plots directly using vuesz without knowing any python. Includes other post processing routines
as well for various data strucure parsing such as Keysight, Agilent, OrbitFR, and Rhode and Schwarz Test Equipment

# Files these scripts can currently reduce

|               File Description               | Extennsion 1 | Extension 2 |  Extension 3  |                                                                                                                                                    Status                                                                                                                                                    |
|:--------------------------------------------:|:------------:|:-----------:|:-------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                  FITS files                  |     .fit     |    .fits    |   .fits.gz    | FITS_AutoPlot 0.2.0: astropy reader, native .vszh5 output (no Veusz needed to write), and an interactive pyqtgraph Plots tab (linked x axes, crosshair with values at the cursor, MJD as UTC dates, CSV/PNG/SVG export). |
| Rhode & Schwarz Network Analyzer ASCII Files |     .sft     |             |               |                                                                                   Working well with averages also being created during data set creation. May look into extrema data set creation and error bars as well.                                                                                    |
|            Comma Separated Files             |     .csv     |             |               |                                                                                                                   Fully Implemented through Delimiter separated Files. Needs full testing.                                                                                                                   |
|         GBO Outdoor Range Data Files         |     .atr     |             |               |                                                                        Working well with polar plots. May want to look at averaging patterns or normalizing during processing. This can be completed by native Veusz plugins as well.                                                                        |
|               Touchstone Files               |     .s1p     |    .s2p     | .s\*p or .snp | Touchstone_AutoPlot 0.5.0: magnitude / phase (wrapped or unwrapped), scikit-rf time domain and time gating (auto or manual, band-pass or band-stop) shown in time and frequency, Smith charts (PNG/SVG/PDF/HTML, bookmarked combined PDF), native .vszh5 output without Veusz. Checked on the 223 example .s2p files (220 processed; 3 are malformed and are reported as failures); the projects load and render in Veusz 4.2.1. |
|        Franks timing data (FranksProcessed)  |   .t00new    | time_gbt.dat |               | Franks_AutoPlot 0.2.0: native .vszh5 output and the same pyqtgraph Plots tab as FITS AutoPlot. |
|        ORBIT/FR (MVG 959 Spectrum) data      |              |             |               | Not written yet: the vendor COM API is not public. See ORBIT_FR/ORBIT_FR_API_REQUIREMENTS.md and run ORBIT_FR/orbitfr_com_probe.py on the 959 Spectrum PC. |

# postProcessing_Launcher.py

GUI for launching the various scripts / tools utilized.
This should be your starting point if you are just running the scripts for plotting data and not developing.

Version 0.2.0 (2026-11-05): one button and one Applications-menu entry (Ctrl+1 ...) per tool, including Franks_AutoPlot; View > Light / Dark mode; Help > About (author, NRAO/GBO, licence) and Veusz plugin instructions; per-launch log files (File > Open Launch Log Folder) with a report if a tool exits with an error at start-up. Run `python postProcessing_Launcher.py --list` to see which tools it found. A frozen (PyInstaller) launcher needs `GBO_POSTPROC_PYTHON` set to the python.exe of the plotting environment. Details: RELEASE_launcher-0.2.0_touchstone-0.5.0.md.

# FITS-AutoPlot.py

Import and autoplot in Veusz (either installed via Python or by system installation method) data within a FITS file.
This has not been rigorously tested, in fact only the Veusz native import has been tested at all.

# GBO_antenna_postProcessing.py

The intent of this script is to be a combined data reduction tool with a Qt GUI.
The current embodiment (branch 0.0.1) is far from the intent but does include the atr data reduction functions.
In the end, it should import various scripts for each file type rather than have all classes and functions within
itself.
Therefore, the atr reduction should come out to its own script in future work.
This has been replaced by postProcessing_Launcer for the time being.

# RAndS_FSW_ASCII_Plotter.py

This script will import and autoplot all data within a single *.sft file, ASCII format, originating from a Rhode &
Schwarz spectrum analyzer.

# fastest_ascii_import.py

This one is ready and works. It is just a single function that can parse any ASCII file with given line numbers and/or
search strings. It then returns a dict with all extracted information.
It also chooses the fastest method based on file size for reading in the data.

# Revisions

Symantic Revisioning with the following
structure: [external or production release].[internal release for review].[draft or work in progress relases / updates]

## Revision By number

Many updates are applied and well documented through GIT commits. If you have any questions, please contact the
developer or submit an issue request.

# Required Python Packagse

While the following is more than is required for running the scripts, it is the suggested environmental packages.
Recall that this is designed for **Python Version 3.8.***

````bash
spyder-kernels==3.0.5 ^
veusz ^
matplotlib ^
xarray ^
astropy ^
qtpy ^
pyside6 ^
hdf5 ^
pdir2 ^
rich ^
pandas ^
pywin32 ^
sympy ^
mccabe ^
numpy ^
dataclasses ^
typing ^
scikit-rf ^
scipy ^
multiprocess ^
pyopencl ^
pathlib ^
cupy ^
psutil ^
qt6-main
````

A [detailed google doc is available for Python Environment Creation Direction](<https://docs.google.com/document/d/1roQzx02ZDnD8I1MvyUEJWbL53ArOPD9zG1YbmjY1jFs/view?usp=sharing>)
