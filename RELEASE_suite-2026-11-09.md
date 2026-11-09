# FitsAutoPlot suite — release notes 2026-11-09

Covers the work of 2026-11-05 to 2026-11-09 on branch `claude/launcher-0.2.0-touchstone-0.5.0`
(local only, not pushed). It includes the time-domain review, pyqtgraph plots in every tool,
light/dark everywhere, and the user's Veusz writer used for every `.vszh5`. It also covers
the 2026-11-09 request: Veusz date-times shown in local ISO 8601, plot names like the
original scripts, and source headers in the page notes. Smith charts in Veusz through the
writer (0.0.26) came last, after these files were posted (§ 6).

![Plots tab in every tool, light and dark](docs/suite_2026-11-09_plots.png)

*Left to right, top to bottom: ATR polar (dark), CSV/TSV (light), R&S zero-span time axis
(dark), Touchstone time domain with gate (dark), Touchstone Smith chart (light), FITS
dt_labels page rendered by Veusz 4.2.1 with the local ISO 8601 axis.*

## 1. Versions

| File | Before | Now | What changed (details in each file's `# %%%%` history) |
|---|---|---|---|
| `vszh5_writer.py` | — (0.0.24 on 11-05) | **0.0.26** | Widget names from the data, `header_notes`/`set_notes`, `local_time_from_veusz`, `display_zone` / `VSZH5_DISPLAY_TZ`, `style_datetime_axis`; 0.0.26 Smith charts |
| `_autoplot_native.py` | 0.1.0 | **0.4.0** | Writes through vszh5_writer; `SetDataLocalTime`, `mark_local_time_axis`, unnamed widgets named after their data when saved; 0.4.0 `add_smith_page` |
| `_autoplot_common.py` | 0.1.0 | **0.3.0** | Suite theme; dt_labels axes as Veusz date-times in local ISO 8601; broken-axis gaps on the same clock |
| `_autoplot_pyqtgraph.py` | 0.1.0 | **0.3.0** | Polar, Smith, log axes, gate region; local ISO 8601 cursor read-out; 0.3.0 project Smith pages drawn as Smith charts |
| `_suite_gui.py` | — | **0.1.0** (new) | Shared theme, menus, About, Open in Veusz, Export PDF/PNG |
| `FITS_AutoPlot.py` / `Franks_AutoPlot.py` | 0.2.0 | **0.3.0** | Date-time datasets, no per-point date text, legacy `_dt` page off by default; Franks NameError fixed |
| `FITS_AutoPlot_plugin.py` / `Franks_AutoPlot_plugin.py` | 0.1.0 | **0.2.0** | Theme styles only the plugin dialog; dt defaults as above |
| `ATR_AutoPlot.py` | 0.1.0 | **0.2.1** | Native writer, polar Plots tab, parser fixes, measurement time as a date-time, header notes |
| `CSV_TSV_AutoPlot.py` | 6.0.0 | **6.1.1** | Native writer, row alignment fix, date-time X columns in local ISO 8601, header notes |
| `RAndS_FSW_ASCII_Plotter.py` | 1.0.3 | **1.1.1** | Zero-span time axis, binary sections, power averages in mW, original page names, sweep times as date-times, notes |
| `Touchstone_AutoPlot.py` | 0.5.0 | **0.7.0** | pyqtgraph previews and Plots tab; 0.4.x graph names, traces named after their data, header notes; 0.7.0 Smith chart pages in the project |
| `Hanging_Threads.py` | 0.0.2 | **0.0.3** | Stoppable, no auto-start, one report per hang |
| `fastest_ascii_import.py` | 0.0.2 | **0.0.3** | Any byte decodes; strict data-line detection (CRLF kept) |
| `veusz_console_scripts_20250623.py` | 0.0.2 | **0.0.3** | `select_file` works (missing tkinter names) |
| `postProcessing_Launcher.py` | 0.2.0 | 0.2.0 | Not changed (no new tools to add) |
| `tests/test_writer_names_time.py` | — | **0.0.1** (new) | Names, notes, local date-time axes |
| `tests/test_smith_chart.py` | — | **0.0.1** (new) | Smith grid geometry, conversions, page scripts, widget tree, Plots-tab recognition |

## 2. Date-times (2026-11-09 request)

**Stored.** Every time stamp is now a Veusz **date** dataset in UTC. FITS/Franks
dt_labels used plain numeric seconds before. The tools store a second date dataset,
`<name>_local`, beside it. The pages draw the local one, so a reader of the file
always has both.

**Shown.** The tick format is `%VDY-%VDm-%VDdT%VDH:%VDM:%VDS-05:00`. That gives
`2026-11-09T08:39:00-05:00`: ISO 8601, 24-hour clock, with the UTC offset of the
computer doing the processing. The axis label reads `Local time (ISO 8601, UTC-05:00)`
and the tick labels are rotated 45°. Veusz's `%VD` format codes have no time-zone field,
so the offset is written into the format as text. That works only while one offset
covers the data. When the data cross a daylight-saving change, the ticks drop the
offset and the axis label lists each offset and when it starts. The writer's design,
since 0.0.24, never prints a wrong offset.

**Changing the zone.** Set `VSZH5_DISPLAY_TZ` to `America/New_York`, `-05:00` or `UTC`
to process for another zone. This is useful on a build server or a container that
runs in UTC. A fixed offset also works on Python 3.8; an IANA name needs Python 3.9 or
later (`zoneinfo`).

**No per-point labels.** The per-point text datasets are no longer written
(`EMIT_TEXT_X_DATASETS = False`). The older labelled `_dt` page is off by default
(check box "Legacy: per-point date text labels page (_dt)").

**Files with no zone.** The ATR "Time and Date" header and the R&S sweep time stamps
are stored as written (`<file>_datetime`, `<file>_sweep_time`), because the files do
not say which zone the instrument clock used. CSV time stamps with an offset or `Z`
are exact. Stamps without one are read as UTC by default, or as local time with
"Times without a zone" / `--naive-time local`.

### Conversions used

The Veusz date origin is 2009-01-01T00:00:00 and the Unix origin is
1970-01-01T00:00:00. The gap between them is 39 years, 10 of which are leap years
(1972, 1976, …, 2008):

$$
\Delta_{09} = -\left(39\cdot 365 + 10\right)\,\text{d}\cdot 86\,400\ \tfrac{\text{s}}{\text{d}}
            = -14\,245 \cdot 86\,400\ \text{s} = -1\,230\,768\,000\ \text{s}
$$

so for a Unix time $t_u$ (UTC seconds) the stored and displayed Veusz values are

$$
v_{\text{UTC}} = t_u + \Delta_{09}, \qquad
v_{\text{local}} = t_u + o(t_u) + \Delta_{09},
$$

where $o(t_u)$ is the UTC offset at that instant: $-18\,000$ s for EST and $-14\,400$ s
for EDT. It is looked up per sample, once per distinct UTC hour. For FITS/Franks MJD
(UTC days), the Unix origin is MJD 40587 (JD 2 440 587.5 − 2 400 000.5), so

$$
t_u = \left(\text{MJD} - 40\,587\right)\cdot 86\,400\ \text{s}.
$$

## 3. Names and page notes (2026-11-09 request)

| Tool | Page names | Graph / trace names | Page notes (Veusz: Properties › notes) |
|---|---|---|---|
| FITS / Franks | Unchanged (as the original scripts) | Unchanged | FITS header cards, unchanged |
| ATR | As the 2025 script | Single frequency: `<file>`; multi-frequency: the y dataset (`<file>_1200MHz_mag`) | `Source:`, header lines, frequencies, measured time (ISO 8601); overlays list every file |
| CSV/TSV | Per file and overlay, as 6.0.0 | Trace = y dataset | `Source:` and the lines above the column header; the overlay lists every file |
| R&S | `<file>::Section N;<stamp>` (underscores as spaces) and `<file> average`, as the 2025 script | Graph = y dataset (`<file>_Section3`, `<file>_avg_dB`, `Averages_avg_dB`) | `Source:`, full header, sweep count. A section page adds its own time; the average page lists every sweep time |
| Touchstone | As 0.5.0 | `<file>_<Sij>_TG_Graph` / `_TG_Freq_Graph` (0.4.x names); trace = y dataset (`<file>_S21_mag_dB`) | Plot title, `Source:`, comment/option lines, summary (ports, points, span, z0) |
| Writer helpers | Caller's name | `g_<y>` / `g_<page>`, xy = y dataset, `img_<ds>`, `cbar_<ds>`, `bars_<first>` | `notes=` on every helper |

`_autoplot_native` names any widget added without a name after the dataset it draws when
the document is saved. `widget_name()` turns any text into a valid Veusz widget name: it
replaces `/`, trims surrounding blanks and rejects `.`/`..`.

**Compatibility note for other programs.** The writer's page helpers used to name the
graph `graph`. A caller that sets `"graph/x/…"` by path after calling a helper must now
use the returned name, or pass `graph_name="graph"`. The AB Power Meter Monitor 1.7.0
and MHM2010 workbench test suites pass with 0.0.25. The workbench's own test pins the
version string `0.0.24`, a literal and not a behaviour difference.

## 4. Time-domain review (2026-11-05)

| Where | Finding | Fix |
|---|---|---|
| R&S `.sft` zero span (x-Unit `s`) | Drawn against Start…Stop as if it were a frequency sweep | Time axis 0…SWT, "Time (s)"; own `AllImported_time` / `Averages_time` pages |
| R&S x-Axis `LOG` | Points spaced linearly | Logarithmic spacing and a log x axis |
| R&S averages | "Average only" made no plot; averages must not be means of dB | Power mean in mW (`linear_power_mean_dbm`, NaN-aware), a page per file plus `Averages` |
| R&S binary sections | `ngRADAR_003.sft` unreadable | `SectionN;stamp\n` + N × float32 LE read |
| FITS/Franks overlay dt pages | `detect_time_breaks_unit_aware` not imported (NameError swallowed): no broken axis on overlays | Imported |
| FITS/Franks broken axes | Gap pairs in UTC while the axis now shows local time | `mjd_break_pairs_to_dtsec(local=True)` |
| FITS/Franks dt pages | Time stored as plain numbers | Veusz date datasets (UTC + local) |
| pyqtgraph date axes | UTC dates | `DateAxisItem(utcOffset=0)` on already-local values, cursor read-out with offset |
| Touchstone TD | Checked against scikit-rf in 0.5.0 (vectorized transform equal to `Network.windowed().s_time`; `skrf.time.time_gate`) | Kept; preview moved to pyqtgraph with a draggable gate |
| CSV date columns | Text sent to Veusz; X and Y dropped NaN separately (rows misaligned) | Columns classified numeric / date-time / text; rows stay aligned |
| ATR azimuth | Fixed 1° step | From the header limits and the point count |

## 5. Found while checking this release

* **R&S notes grew as sweeps².** Every section page listed every sweep time, on the page
  and on the graph. With all example files (588 pages), the document script was 11.5 MB
  and Veusz 4.2.1 had not finished loading it after 5 minutes. Each section page now
  names only its own sweep. The script is 2.2 MB and loads in 13.8 s.

## 6. Smith charts in Veusz (writer 0.0.26, Touchstone 0.7.0)

![Veusz Smith charts written by vszh5_writer 0.0.26](docs/smith_veusz_writer_0.0.26.png)

*Rendered by Veusz 4.2.1 from a file written without Veusz. Left: an impedance chart with
VSWR circles and two traces, legend beside the chart. Right: an admittance chart.*

**Method.** Veusz has no Smith-chart widget. The chart is assembled from widgets it does
have, so it stays editable in Veusz:

* The graph has a fixed aspect ratio of 1. Its axes are $\mathrm{Re}\,\Gamma$ and
  $\mathrm{Im}\,\Gamma$, hidden, at $\pm 1.14$.
* The grid is ordinary xy lines read from datasets. One xy widget draws a whole family of
  circles, joined with NaN breaks.
* The grid values are point labels of marker-less xy widgets (a text dataset).
* The data are xy traces of $\mathrm{Im}\,\Gamma$ against $\mathrm{Re}\,\Gamma$.
* The graph notes say "Smith chart (impedance)" or "(admittance)" and give $Z_0$. That is
  how the Plots tab recognises the page and draws it with its own interactive Smith plot,
  which reads out f, Γ, Z and VSWR at the cursor.

**Grid equations.** With the normalized impedance $z = Z/Z_0 = r + jx$,

$$
\Gamma = \frac{z-1}{z+1} = \frac{(r^2 + x^2 - 1) + j\,2x}{(r+1)^2 + x^2}.
$$

Eliminating $x$ (constant $r$) gives the resistance circles; eliminating $r$ (constant $x$)
gives the reactance circles:

$$
\left(\Gamma_r - \frac{r}{1+r}\right)^2 + \Gamma_i^2 = \left(\frac{1}{1+r}\right)^2,
\qquad
\left(\Gamma_r - 1\right)^2 + \left(\Gamma_i - \frac{1}{x}\right)^2 = \left(\frac{1}{x}\right)^2 .
$$

Only the part of each reactance circle inside $|\Gamma| \le 1$ ($r \ge 0$) is drawn. It is
traced exactly as $\Gamma(r)$ with $r = \tan\theta$, $0 \le \theta < \pi/2$, so every arc starts
on the unit circle and ends at $\Gamma = 1$. Setting $r = 0$ gives $\Gamma = e^{j\varphi}$ with
$\varphi = 2\arctan(1/x)$, which is where the $x$ labels go.

The admittance chart uses $y = Y/Y_0 = g + jb$ and
$\Gamma = (1-y)/(1+y) = -\,(y-1)/(y+1)$: the same grid rotated by 180°, as scikit-rf draws it.

A VSWR circle $S$ has radius $|\Gamma| = (S-1)/(S+1)$. The defaults are 1.5, 2, 3 and 5,
scikit-rf's values.

**Using it.**

```python
import vszh5_writer as w
grid = w.smith_grid("smith_z", "z", vswr=w.SMITH_VSWR)          # once per document
data = w.smith_trace_datasets("dut_S11_gamma", gamma, freq_hz)  # _re, _im, _freq_Hz
doc = w.VszDoc("my tool")
w.smith_page(doc, "Smith_dut", [w.smith_trace("dut_S11_gamma", "S11")], grid,
             title="dut - Smith chart (Z0 = 50 ohm)", notes=w.header_notes(header, source=path))
with w.VeuszWriter("out.vszh5") as vw:
    grid.write(vw)
    for name, values in data.items():
        vw.dataset_1d(name, values)
    vw.finish(doc.text())
```

On a `NativeDocument` (every AutoPlot tool), use `_autoplot_native.add_smith_page(doc.Root, ...)`
with the same grid and traces.

**Touchstone AutoPlot 0.7.0.** "Smith chart pages in the Veusz project" is on by default in
the Smith settings. Prepare Smith Charts then adds a page `Smith_<file>` per file:

* It shows the reflection parameters (S11, S22, …) at the reference impedance.
* It uses the chosen chart type, labels and VSWR circles.
* The graph is `<file>_Smith_Graph`.
* Each trace is named after its dataset `<file>_<Sii>_gamma_im`.

Transmission parameters are left out of these pages. They are still in the PNG/SVG/PDF/HTML
Smith export.

**Checks.**

* `tests/test_smith_chart.py` (7 tests): every grid point satisfies the circle equations,
  and $z$ recovered from the grid has the right $r$ or $x$; Γ ↔ Z ↔ VSWR round trips; the
  page script; the widget tree; recognition by the Plots tab.
* Veusz 4.2.1 renders the demo and the Touchstone pages (all 6 pages of a 3-file project, PASS).
* Every setting used (graph `aspect`, axis `hide`, xy `labels` with
  `Label/posnHorz|posnVert`, key `horzManual`) also exists in the Veusz 3.6.2 source.
* The Smith code also works with pyqtgraph 0.13.3 and PyQt5 (the Python 3.8 stack).
* The writer's existing page helpers write byte-identical scripts to 0.0.25.

## 7. Verification

| Check | Result |
|---|---|
| `python -m unittest discover -s tests` (`test_native_export.py` 16, `test_writer_names_time.py` 5, `test_smith_chart.py` 7) | 28 tests OK |
| Headless GUI runs, `VSZH5_DISPLAY_TZ=America/New_York`: FITS, Franks, CSV/TSV, ATR, R&S, Touchstone (Plots tab, light and dark, save) | Pass |
| Veusz 4.2.1 load and render of each tool's `.vszh5` (PDF/PNG/SVG) | Pass: ISO ticks `…T08:39:00-05:00`, label with offset, notes present, names as tabulated |
| R&S all example files, mode 0 (588 pages, 594 datasets), `tests/validate_in_veusz.py` | Loads in 13.8 s; all 588 pages rendered to SVG (PASS) |
| Writer 0.0.25 and 0.0.26 as a drop-in: AB Power Meter Monitor 1.7.0 runner (all 11 suites: appender, offline renders in Veusz, memory release, file locks, exporter, Grafana, IP separation, live plots, memory, GUI soak, 10-min headless soak) and MHM2010 workbench suite (112 tests) | Pass; the only failure is the workbench test that pins the writer version string |
| Python 3.8 grammar (`ast.parse(feature_version=(3, 8))`) of the 20 changed or new `.py` files | Pass |

## 8. Data gaps and limits

* There is no Python 3.8 interpreter, Windows machine or real GBO FITS file here. The
  Windows paths, the Veusz installer path and the FITS pages were checked only with
  synthetic data.
* The ATR and R&S time stamps carry no zone. They are stored as the instrument wrote
  them; please confirm whether the range PC and the analyzers run on local time or UTC.
* For ATR multi-frequency files, the order of the frequency lines is assumed to be
  lower → upper.
* For R&S zero span, the x axis is taken as 0…SWT from the header.
* CSV time stamps without a zone are assumed UTC unless the option says local.

## 9. References

| # | Reference (IEEE) |
|---|---|
| [1] | Veusz developers, *Veusz 4.2.1* source (`document/loader.py`, `utils/dates.py`, `widgets/axisticks.py`). [Online]. Available: https://pypi.org/project/veusz/4.2.1/ |
| [2] | Veusz developers, "Reading data — dates," *Veusz manual*. [Online]. Available: https://veusz.github.io/docs/manual/datasets.html |
| [3] | *Date and time — Representations for information interchange — Part 1: Basic rules*, ISO 8601-1:2019. [Online]. Available: https://www.iso.org/standard/70907.html |
| [4] | P. Ganssle, "PEP 615 — Support for the IANA Time Zone Database in the Standard Library," Python Software Foundation, 2020. [Online]. Available: https://peps.python.org/pep-0615/ |
| [5] | W. D. Pence, L. Chiappetti, C. G. Page, R. A. Shaw, and E. Stobie, "Definition of the Flexible Image Transport System (FITS), version 3.0," *Astron. Astrophys.*, vol. 524, A42, 2010, doi: 10.1051/0004-6361/201015362. [Online]. Available: https://www.aanda.org/articles/aa/full_html/2010/16/aa15362-10/aa15362-10.html |
| [6] | IBIS Open Forum, *Touchstone® File Format Specification, Version 2.1*, ratified Jan. 26, 2024. [Online]. Available: https://www.ibis.org/touchstone_ver2.1/touchstone_ver2_1.pdf |
| [7] | A. Arsenovic *et al.*, "scikit-rf: An open source Python package for microwave network creation, analysis, and calibration," *IEEE Microw. Mag.*, vol. 23, no. 1, pp. 98–105, Jan. 2022, doi: 10.1109/MMM.2021.3117139. |
| [8] | Agilent Technologies, "Time domain analysis using a network analyzer," Application Note 1287-12, 5989-5723EN. [Online]. Available: https://www.keysight.com/us/en/assets/7018-01451/application-notes/5989-5723.pdf |
| [9] | pyqtgraph developers, "DateAxisItem." [Online]. Available: https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/dateaxisitem.html |
| [10] | h5py developers, "h5py 3.16.0." [Online]. Available: https://pypi.org/project/h5py/ |
| [11] | P. H. Smith, "Transmission line calculator," *Electronics*, vol. 12, no. 1, pp. 29–31, Jan. 1939. [Online]. Available: https://rfcafe.com/references/electronics-mag/transmission-line-calculator-phillip-h-smith-electronics-magazine-january-1939.htm |
