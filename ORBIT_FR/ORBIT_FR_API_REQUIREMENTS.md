---
title: ORBIT/FR plotter - what is needed before it can be written
author: William W. Wallace (NRAO/GBO)
date: 2026-11-05
version: 0.0.1
status: Staged. No plotter yet: the vendor API reference is not public.
---

# ORBIT/FR plotter: what is needed

**Status.** `ORBITFR_AutoPlot.py` has **not** been written. MVG (ORBIT/FR) confirms that 959 Spectrum can be scripted through COM, but the object model (class names, methods, data structures) is not published anywhere I could find. It comes with the licensed software. A plotter written now would have to guess method names, so it would not run.

What is here instead:

| File | Purpose |
|---|---|
| `ORBIT_FR_API_REQUIREMENTS.md` | This document: what is known, what is missing, and the plan |
| `orbitfr_com_probe.py` 0.0.1 | Read-only Windows tool that finds the 959 Spectrum COM classes and type libraries on the measurement PC and writes them to a Markdown report. It never creates a COM object, so it cannot start 959 Spectrum or move a positioner |

The launcher (`postProcessing_Launcher.py` 0.2.0) already has an entry for `ORBIT_FR/ORBITFR_AutoPlot.py`. Its button and Applications-menu item **appear automatically once that file exists**, so nothing is shown as "under development" in the meantime.

## 1. What the public sources confirm

| Topic | What MVG states | Source | Confidence |
|---|---|---|---|
| Scripting technology | 959 Spectrum scripting is "based on MS COM technology" | [2] | High (vendor product page) |
| Script hosts | "All scripting environments that support COM can be used"; test parameter files initialise an acquisition, and scripts program the changes | [1] | High (vendor release note, v7) |
| Software modules | 959 Spectrum combines 959 Acquisition and 959 DataPro (NF-FF transforms, beam peaks, quick-look plots, device tables, echo reduction) | [2] | High |
| Controller API | OFR 9800A switching controller: LAN, TCP/IP, "high-level command language (HLCL)", and "support for third-party application development (API and COM)" | [3] | High (vendor datasheet, 2019) |
| COM object model (ProgIDs, objects, methods) | Not published | none | Gap |
| DataPro export file formats | Not published | none | Gap |
| 959 Spectrum version at GBO | Unknown | none | Gap |

The ORBIT/FR "959 Spectrum Readme" page (orbitfr.com) blocks automated access, so I could not read it.

A Python script reaches COM through **pywin32** (`win32com.client`). That is standard Windows practice and needs no vendor library beyond 959 Spectrum itself.

## 2. What I need from you

Items 1 to 3 are enough to start. Items 4 to 6 settle the design.

| # | Item | How to get it | Why |
|---|---|---|---|
| 1 | **The COM discovery report** | On the 959 Spectrum PC: `python orbitfr_com_probe.py` (needs pywin32: `mamba install pywin32`). Send back the `orbitfr_com_report_*.md` it writes | Gives the real ProgIDs, objects, methods and arguments. Everything else is built on this |
| 2 | **959 Spectrum scripting / automation manual** | Usually installed with 959 Spectrum (Help folder or Start-menu documentation), or from MVG support | Explains call order, units and side effects that a type library does not show |
| 3 | **Two or three example data files**, plus one exported ASCII file if DataPro can export | Any recent far-field cut and a frequency sweep | The file reader is written and tested against these |
| 4 | 959 Spectrum version and bitness (32- or 64-bit), Windows version | Help > About in 959 Spectrum | A 32-bit in-process COM server cannot load into 64-bit Python; an out-of-process (.exe) server can |
| 5 | Measurement types to plot | Your call | e.g. far-field cuts, NF raw or transformed, frequency sweeps, multi-port or multi-probe, co-pol and cross-pol |
| 6 | **Read-only, or should it also run acquisitions?** | Your call | A read-only plotter cannot move hardware. Acquisition control needs interlocks and confirmation dialogs and is a much larger job |

## 3. Proposed design (once the API is known)

The design follows the other tools in this repository, so nothing new needs installing apart from pywin32.

| Layer | Plan |
|---|---|
| Data source A: files | Reader for 959 data and DataPro ASCII exports; works on any PC, no 959 Spectrum needed. The first target, because it is testable |
| Data source B: live COM | `win32com.client.Dispatch(<ProgID from the report>)`, **read-only calls only**: open a data set, list cuts, frequencies and polarizations, fetch arrays. Off by default and greyed out when pywin32 or 959 Spectrum is missing |
| Model | NumPy arrays per cut: angle (deg), frequency (Hz), amplitude (dB), phase (deg), plus polarization and metadata. Large sets are streamed per cut and released after writing, with an optional memmap or pickle off-load, so RAM stays bounded |
| Interactive plots | pyqtgraph: rectangular and polar cuts, overlays across frequency, cursor readout. Same look as the FITS and Franks AutoPlot pyqtgraph update |
| Output | Native `.vszh5` through `_autoplot_native.NativeDocument`: one page per cut, plus overlay and polar pages. Veusz is needed only to view |
| Derived values | −3 dB and −10 dB beamwidths, peak gain and angle, first sidelobe level, cross-pol discrimination. Vectorised NumPy, optional CuPy, files processed in parallel |
| GUI | qtpy with PySide6; File / View (Light / Dark) / Help > About (NRAO) menus, as in the other tools |
| Launcher | Save as `ORBIT_FR/ORBITFR_AutoPlot.py` and it appears in `postProcessing_Launcher.py` |

## 4. Open questions

1. Is the GBO outdoor range's `.ATR` format an ORBIT/FR export, or GBO's own? The example `.ATR` headers ("Sweep by Position CCW", azimuth limits, power level) do not identify the software.
2. Should ORBIT/FR results be overlaid with ATR patterns on the same Veusz pages, for range-to-range comparison?
3. Which ORBIT/FR controller is at GBO (for example the AL-4806 family, or an OFR 9800A)? This matters only if live control is ever wanted.

## References

| # | Reference (IEEE) |
|---|---|
| [1] | MVG (Microwave Vision Group), "MVG releases AM software upgrade for the US market: 959 SPECTRUM v7," news release. [Online]. Available: https://www.mvg-world.com/en/news/mvg-releases-am-software-upgrade-for-the-us-market-959-spectrum-v7 (accessed Nov. 5, 2026). |
| [2] | MVG, "959 Spectrum," antenna measurement software product page. [Online]. Available: https://www.mvg-world.com/de/produkte/antennenmessung/software/959-spectrum (accessed Nov. 5, 2026). |
| [3] | MVG, "OFR 9800A High Speed Switching Controller - Measurement Controller," datasheet, 2019. [Online]. Available: https://www.mvg-world.com/media/954/download/reference (accessed Nov. 5, 2026). |

## Revision history

| Version | Date | Change |
|---|---|---|
| 0.0.1 | 2026-11-05 | First internal draft. Requirements, design and the read-only COM probe |
