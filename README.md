# FitsAutoPlot
Python 3.8 based scripts for autoplotting FITS files using Veusz and astropy. This allows any user to view the final data sets and alter the plots directly using vuesz without knowing any python. Includes other post processing routines as well for various data strucure parsing such as Keysight, Agilent, OrbitFR, and Rhode and Schwarz Test Equipment

# Files these scripts can currently reduce


| File Description | Extennsion 1 | Extension 2| Extension 3 |
| :----: | :----: | :----: | :----: |
| FITS files | .fit | .fits | |
| Rhode & Schwarz Network Analyzer ASCII Files | .sft | | |
| Comma Separated Files | Not yet Implemented | | |
| GBO Outdoor Range Data Files | .atr | | |
| Touchstone Files | .s1p | .s2p | |

# postProcessing_Launcher.py
This is the main launching GUI. I have noticed on windows 11 it takes a inordinate amount of time to launch each option.
This will need some troubleshooting in the future.
If you just want to use the program / scripts, start by executing this file.

## Classes
Will update in future releases...

## Functions
Will update in future releases...

# FITS-AutoPlot.py
Import and autoplot in Veusz (either installed via Python or by system installation method) data within a FITS file.
This has not been rigorously tested, in fact only the Veusz native import has been tested at all.

## Classes
Will update in future releases...

### Functions
Will update in future releases...
# GBO_antenna_postProcessing.py
The intent of this script is to be a combined data reduction tool with a Qt GUI.
The current embodiment (branch 0.0.1) is far from the intent but does include the atr data reduction functions.
In the end, it should import various scripts for each file type rather than have all classes and functions within itself.
Therefore, the atr reduction should come out to its own script in future work.

## Classes
Will update in future releases...
### Functions
Will update in future releases...

# RAndS_FSW_ASCII_Plotter.py
This script will import and autoplot all data within a single *.sft file, ASCII format, originating from a Rhode & Schwarz spectrum analyzer.

## Classes
Will update in future releases...

### Functions
Will update in future releases...

# fastest_ascii_import.py
This one is ready and works. It is just a single function that can parse any ASCII file with given line numbers and/or search strings. It then returns a dict with all extracted information.
It also chooses the fastest method based on file size for reading in the data.

## Classes
Will update in future releases...

### Functions
Will update in future releases...

# Revisions
Symantic Revisioning with the following structure: [external or production release].[internal release for review].[draft or work in progress relases / updates]


Building the scripts and uploading the starting point. As of this version it is not yet fully working. I am just getting the repo setup and code added to organize as needed in later revisions. All such changes will be tracked in branches with the release version if so working.
