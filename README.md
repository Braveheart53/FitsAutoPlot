# FitsAutoPlot
**Python 3.8** based scripts for autoplotting FITS files using Veusz and astropy. This allows any user to view the final data sets and alter the plots directly using vuesz without knowing any python. Includes other post processing routines as well for various data strucure parsing such as Keysight, Agilent, OrbitFR, and Rhode and Schwarz Test Equipment

# Files these scripts can currently reduce


| File Description | Extennsion 1 | Extension 2| Extension 3 | Status |
| :----: | :----: | :----: | :----: | :----: |
| FITS files | .fit | .fits | | Partially working. Imports via Veusz Native Works and is tested, importing via AstroPy processing prior to veusz dataset creation not yet tested. |
| Rhode & Schwarz Network Analyzer ASCII Files | .sft | | | Working well with averages also being created during data set creation. May look into extrema data set creation and error bars as well. |
| Comma Separated Files | .csv | | | Not yet implemented fully into GUI format(s). Need to update. |
| GBO Outdoor Range Data Files | .atr | | | Working well with polar plots. May want to look at averaging patterns or normalizing during processing. This can be completed by native Veusz plugins as well. |


# postProcessing_Launcher.py
GUI for launching the various scripts / tools utilized. 
This should be your starting point if you are just running the scripts for plotting data and not developing.

# FITS-AutoPlot.py
Import and autoplot in Veusz (either installed via Python or by system installation method) data within a FITS file.
This has not been rigorously tested, in fact only the Veusz native import has been tested at all.


# GBO_antenna_postProcessing.py
The intent of this script is to be a combined data reduction tool with a Qt GUI.
The current embodiment (branch 0.0.1) is far from the intent but does include the atr data reduction functions.
In the end, it should import various scripts for each file type rather than have all classes and functions within itself.
Therefore, the atr reduction should come out to its own script in future work.
This has been replaced by postProcessing_Launcer for the time being.


# RAndS_FSW_ASCII_Plotter.py
This script will import and autoplot all data within a single *.sft file, ASCII format, originating from a Rhode & Schwarz spectrum analyzer.

# fastest_ascii_import.py
This one is ready and works. It is just a single function that can parse any ASCII file with given line numbers and/or search strings. It then returns a dict with all extracted information.
It also chooses the fastest method based on file size for reading in the data.


# Revisions
Symantic Revisioning with the following structure: [external or production release].[internal release for review].[draft or work in progress relases / updates]

## Revision By number
Many updates are applied and well documented through GIT commits. If you have any questions, please contact the developer or submit an issue request.

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

A [detailed google doc is available for Python Environment Creation Direction](<https://docs.google.com/document/d/1roQzx02ZDnD8I1MvyUEJWbL53ArOPD9zG1YbmjY1jFs/edit?usp=sharing>)
