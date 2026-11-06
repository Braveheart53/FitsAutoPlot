# -*- coding: utf-8 -*-
"""
=============================================================================
_autoplot_pyqtgraph.py
-----------------------------------------------------------------------------
# %% Header Info

Interactive pyqtgraph plots of an AutoPlot project, for the "Plots" tab of
FITS_AutoPlot and Franks_AutoPlot.  Modelled on the AB Power Meter Monitor
1.7.0 live plots (ab_live_plots.py 0.1.0, same author): the same plot
widget, legend-and-values column, crosshair, wheel rule, colours and theme.

WHAT IT DRAWS
-------------
The panel reads the in-memory ``NativeDocument`` that is saved as the .vszh5
project, so the interactive view always shows exactly the pages, graphs,
traces and colours of the Veusz project:

    [Page: <combo>]  [<] [>]  [x] MJD as UTC date  [Fit all]   hint
    +------------------------------------------+-----------------------------+
    | TracePlot: graph 1 of the page           | legend column               |
    |   every xy trace of that graph           | [x] trace  at cursor  last  |
    +------------------------------------------+-----------------------------+
    | TracePlot: graph 2 ...  (x axes linked, one crosshair for the page)    |

* x axes of one page are linked; a crosshair follows the mouse in every
  plot of the page, and the legend column lists each trace's value at the
  cursor and its last value [P1]-[P3].
* Mouse: drag = pan, right-drag = zoom one axis, Ctrl+wheel = zoom (a plain
  wheel scrolls the page), right-click = pyqtgraph menu (incl. Export: CSV /
  PNG / SVG), "A" corner button = auto range.  Untick a legend row to hide
  that trace.
* x axes: Veusz datetime axes (seconds since 2009-01-01 UTC) and MJD axes
  (optional) are drawn as UTC dates; everything else stays numeric.
  Time gaps longer than GAP_FACTOR x the median step break the line, the
  pyqtgraph counterpart of the project's broken x axes.
* 2-D image graphs (FITS image HDUs) are drawn with an ImageItem and a
  colour bar.

DATA AND MEMORY
---------------
Only the page on screen has plots.  Changing page disposes the previous
page's plots (including pyqtgraph's parentless right-click menus, see
TracePlot.dispose) so memory stays bounded by one page.  The plot arrays are
the project's own arrays (no copy) except where a unit conversion or a gap
break needs one.  pyqtgraph thins every trace to the screen ("peak"
downsampling, clip to view), so million-point columns stay responsive.

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Company: NRAO / Green Bank Observatory (GBO)
Licence: GPL-3.0-or-later
Compatible with: Python 3.8+ (pyqtgraph 0.13.3) and 3.12+ (pyqtgraph 0.14),
                 qtpy with PySide6 or PyQt5

# %%% References
[P1] pyqtgraph developers, "pyqtgraph documentation."
     https://pyqtgraph.readthedocs.io/en/latest/
[P2] pyqtgraph developers, "DateAxisItem."
     https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/dateaxisitem.html
[P3] pyqtgraph developers, "PlotDataItem" (setData, downsampling,
     clipToView, connect="finite").
     https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/plotdataitem.html
[P4] W. W. Wallace, "ab_live_plots.py 0.1.0," AB Power Meter Monitor 1.7.0,
     GBO_Maser_Control repository, 2026.
[P5] J. Sanders, "polar.py" (Polar graph widget), Veusz 4.2.
     https://github.com/veusz/veusz/blob/master/veusz/widgets/polar.py
[P6] P. H. Smith, "Transmission line calculator," Electronics, vol. 12,
     no. 1, pp. 29-31, Jan. 1939.
[P7] D. M. Pozar, Microwave Engineering, 4th ed. Hoboken, NJ, USA: Wiley,
     2012, ch. 2.4 (the Smith chart).

# %%% Revisions
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.2.0: Polar graphs, Smith chart, log axes, project axis ranges.
# Date: 2026-11-05
#              * Veusz 'polar' graphs with 'nonorthpoint' traces (ATR
#                patterns): PolarPlot draws them with the project's
#                minradius/maxradius, units (degrees, radians, fractions,
#                percentages), direction, position0 and log radius, using the
#                same transform as Veusz's polar.py; the legend column gives
#                r at the cursor angle.
#              * SmithPlot: Smith chart (impedance or admittance grid) with a
#                readout of f, Gamma, Z and VSWR at the nearest point, for
#                Touchstone AutoPlot's interactive Smith tab.
#              * TracePlot honours the project's axis log settings and fixed
#                min/max limits (Auto stays auto), and can show a shaded
#                region (time gate) that the user may drag.  On a log x axis
#                the cursor converts the view's log10(x) back to x.
#              * One page in memory at a time, as before.
# %%%% 0.1.0: First version - pyqtgraph view of the AutoPlot project.
# Date: 2026-11-05
#              TracePlot, wheel rule, theme and dispose() follow
#              ab_live_plots.py 0.1.0; data comes from the NativeDocument
#              widget tree (pages > graphs > xy / image) instead of a live
#              store.  Datetime and MJD axes as UTC dates, gap breaks,
#              2-D images, page navigation, one page in memory at a time.
=============================================================================
"""
# %% Import Modules
from __future__ import annotations

import datetime
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("QT_API", "pyside6")   # qtpy and pyqtgraph pick the same binding

import numpy as np  # noqa: E402
from qtpy import QtGui, QtWidgets  # noqa: E402
from qtpy.QtCore import Qt, Signal  # noqa: E402
import pyqtgraph as pg  # noqa: E402

VERSION = "0.2.0"
__version__ = VERSION

# %% Settings
#: Fallback trace colours (Tableau 10, as in the AB monitor) when the project
#: gives none.  Project colours (PlotLine/color) are used when present.
COLOURS: Tuple[str, ...] = ("#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                            "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac")
#: (background, foreground, crosshair) per theme (same values as the AB monitor).
THEMES: Dict[str, Tuple[str, str, str]] = {"dark": ("#1e1e2e", "#cdd6f4", "#9399b2"),
                                          "light": ("#ffffff", "#202020", "#707070")}
PLOT_HEIGHT = 270              # px per plot row
LEGEND_WIDTH = 330             # px of the legend column (user can drag the splitter)
SYMBOL_MAX_POINTS = 200        # point symbols only up to this many points per trace
GAP_FACTOR = 10.0              # a step > GAP_FACTOR x median step breaks the line
WHEEL_NEEDS_CTRL = True        # plain wheel scrolls the page, Ctrl+wheel zooms
VEUSZ_EPOCH_UNIX = 1230768000.0  # 2009-01-01T00:00:00Z: Veusz datetime zero
MJD_UNIX_EPOCH = 40587.0       # MJD of 1970-01-01T00:00:00Z
MJD_PLAUSIBLE = (30000.0, 100000.0)   # values outside are not treated as MJD

_LINE_STYLES = {"solid": Qt.PenStyle.SolidLine, "dashed": Qt.PenStyle.DashLine,
                "dotted": Qt.PenStyle.DotLine, "dash-dot": Qt.PenStyle.DashDotLine}
_MARKERS = {"circle": "o", "square": "s", "diamond": "d", "triangle": "t",
            "cross": "x", "plus": "+", "star": "star"}


# %% Small helpers
def _ctrl_pressed(event: Any) -> bool:
    """True when Ctrl is held during a (scene) wheel event."""
    try:
        return bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
    except Exception:  # pragma: no cover - unusual binding
        return False


def _utc_text(epoch: Optional[float]) -> str:
    """UTC wall-clock text of a Unix time ('-' when unknown)."""
    if epoch is None or not np.isfinite(epoch):
        return "-"
    try:
        stamp = datetime.datetime.fromtimestamp(float(epoch), tz=datetime.timezone.utc)
    except (OverflowError, OSError, ValueError):
        return "-"
    return stamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def _value_text(value: Optional[float]) -> str:
    """Compact number; '-' for None/NaN."""
    if value is None or not np.isfinite(value):
        return "-"
    return "{0:.7g}".format(float(value))


def _align_right(item: Any) -> None:
    """Right-align a table cell (Qt.Alignment on new PySide6, int on old)."""
    alignment = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
    try:
        item.setTextAlignment(alignment)
    except TypeError:  # PySide6 6.2 / PyQt5 int overload
        item.setTextAlignment(int(alignment))


def _swatch(colour: str) -> Any:
    """Small square icon in a trace colour (legend column)."""
    pixmap = QtGui.QPixmap(12, 12)
    pixmap.fill(QtGui.QColor(colour))
    return QtGui.QIcon(pixmap)


def break_gaps(x: np.ndarray, y: np.ndarray,
               factor: float = GAP_FACTOR) -> Tuple[np.ndarray, np.ndarray]:
    """
    Insert one NaN after every step longer than ``factor`` x the median step.

    Only for monotonic increasing x (sorted time series); other data are
    returned unchanged (no copy).  With ``connect="finite"`` the NaN breaks
    the drawn line, as a broken x axis does in the Veusz project.
    """
    if x.size < 3:
        return x, y
    steps = np.diff(x)
    finite = steps[np.isfinite(steps)]
    if finite.size == 0 or np.any(finite < 0):
        return x, y
    positive = finite[finite > 0]
    if positive.size == 0:
        return x, y
    limit = factor * float(np.median(positive))
    cut = np.nonzero(steps > limit)[0]
    if cut.size == 0:
        return x, y
    x_out = np.insert(x.astype(float, copy=False), cut + 1, np.nan)
    y_out = np.insert(y.astype(float, copy=False), cut + 1, np.nan)
    return x_out, y_out


# %% pyqtgraph items: a plain wheel scrolls the page, Ctrl+wheel zooms
def _scroll_enclosing_area(event: Any) -> bool:
    """Scroll the nearest QScrollArea around the plot by the wheel's steps."""
    widget = event.widget() if hasattr(event, "widget") else None
    while widget is not None and not isinstance(widget, QtWidgets.QScrollArea):
        widget = widget.parentWidget()
    if widget is None:
        return False
    bar = widget.verticalScrollBar()
    notches = float(event.delta()) / 120.0
    lines = QtWidgets.QApplication.wheelScrollLines()
    bar.setValue(bar.value() - int(round(notches * lines * bar.singleStep())))
    return True


def _wheel_passes_through(event: Any) -> bool:
    """True when the wheel must NOT zoom: scroll the page instead."""
    if not WHEEL_NEEDS_CTRL or _ctrl_pressed(event):
        return False
    if _scroll_enclosing_area(event):
        event.accept()
    else:
        event.ignore()
    return True


class _ViewBox(pg.ViewBox):
    """ViewBox whose wheel zoom needs Ctrl."""

    def wheelEvent(self, ev: Any, axis: Any = None) -> None:  # noqa: N802 - Qt override
        if not _wheel_passes_through(ev):
            super().wheelEvent(ev, axis)


class _AxisItem(pg.AxisItem):
    """Value axis with the same wheel rule."""

    def wheelEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        if not _wheel_passes_through(event):
            super().wheelEvent(event)


class _DateAxisItem(pg.DateAxisItem):
    """UTC date axis with the same wheel rule."""

    def wheelEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        if not _wheel_passes_through(event):
            super().wheelEvent(event)


class _PlotWidget(pg.PlotWidget):
    """PlotWidget that reports when the mouse leaves it (to clear the crosshair)."""
    left = Signal()

    def leaveEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().leaveEvent(event)
        self.left.emit()


def configure(antialias: bool = True, wheel_needs_ctrl: bool = True) -> None:
    """Global pyqtgraph options and the wheel rule (call before plots exist)."""
    global WHEEL_NEEDS_CTRL
    WHEEL_NEEDS_CTRL = bool(wheel_needs_ctrl)
    pg.setConfigOptions(antialias=bool(antialias))


# %% Reading the project: pages > graphs > traces
def _setting(widget: Any, key: str, default: Any = None) -> Any:
    """A NativeWidget setting ('PlotLine/color', 'label', ...)."""
    return getattr(widget, "settings", {}).get(key, default)


def _children(widget: Any, kinds: Sequence[str]) -> List[Any]:
    return [c for c in getattr(widget, "children", []) if c.kind in kinds]


def _graphs_of(widget: Any) -> List[Any]:
    """Graphs of a page in drawing order (directly or inside grids)."""
    found = []
    for child in getattr(widget, "children", []):
        if child.kind in ("graph", "polar"):
            found.append(child)
        elif child.kind == "grid":
            found.extend(_graphs_of(child))
    return found


def _numeric(doc: Any, name: Any) -> Tuple[Optional[np.ndarray], str]:
    """(array, kind) of a numeric 1-D or date dataset, else (None, kind)."""
    if not isinstance(name, str) or name not in getattr(doc, "datasets", {}):
        return None, ""
    kind, parts = doc.datasets[name]
    if kind not in ("1d", "date"):
        return None, kind
    return parts["data"], kind


def _looks_like_mjd(name: str, label: str, x: np.ndarray) -> bool:
    """MJD axis: 'MJD' in the dataset name or axis label and plausible values."""
    if "mjd" not in (name + " " + label).lower():
        return False
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return False
    return MJD_PLAUSIBLE[0] <= float(np.min(finite)) and float(np.max(finite)) <= MJD_PLAUSIBLE[1]


def document_pages(doc: Any) -> List[Dict[str, Any]]:
    """
    Pages of a NativeDocument as plain descriptions (no Qt objects).

    Returns
    -------
    list of dict
        [{"name", "graphs": [{"name", "title", "x_label", "y_label",
        "x_kind" ("numeric" | "veusz_date" | "mjd"), "traces": [{"label",
        "x_name", "y_name", "colour", "style", "marker", "line"}],
        "images": [dataset names]}]}].  Traces whose x or y dataset is not
        numeric (text label axes) are left out.
    """
    pages = []
    root = getattr(doc, "Root", None)
    for page in _children(root, ("page",)):
        graphs = []
        for graph in _graphs_of(page):
            if graph.kind == "polar":
                polar = _polar_graph(doc, graph)
                if polar is not None:
                    graphs.append(polar)
                continue
            axes = {a.name: a for a in _children(graph, ("axis", "axis-broken"))}
            x_axis, y_axis = axes.get("x"), axes.get("y")
            x_label = str(_setting(x_axis, "label", "") or "") if x_axis else ""
            y_label = str(_setting(y_axis, "label", "") or "") if y_axis else ""
            x_mode = str(_setting(x_axis, "mode", "numeric") or "numeric") if x_axis else "numeric"
            title_widget = next((c for c in _children(graph, ("label",))
                                 if c.name == "plotTitle"), None)
            title = str(_setting(title_widget, "label", "") or "") if title_widget else ""
            traces, x_kinds = [], set()
            for xy in _children(graph, ("xy",)):
                x_name, y_name = _setting(xy, "xData"), _setting(xy, "yData")
                x, x_kind = _numeric(doc, x_name)
                y, _y_kind = _numeric(doc, y_name)
                if x is None or y is None:
                    continue
                if x_mode == "datetime" or x_kind == "date":
                    x_kinds.add("veusz_date")
                elif _looks_like_mjd(str(x_name), x_label, x):
                    x_kinds.add("mjd")
                else:
                    x_kinds.add("numeric")
                traces.append({
                    "label": str(_setting(xy, "key", "") or y_name),
                    "x_name": x_name, "y_name": y_name,
                    "colour": _setting(xy, "PlotLine/color") or _setting(xy, "MarkerFill/color"),
                    "style": _setting(xy, "PlotLine/style", "solid"),
                    "marker": _setting(xy, "marker", "none"),
                    "line": not bool(_setting(xy, "PlotLine/hide", False)),
                })
            images = [str(_setting(img, "data")) for img in _children(graph, ("image",))
                      if _setting(img, "data") in getattr(doc, "datasets", {})]
            if not traces and not images:
                continue
            if len(x_kinds) == 1:
                x_kind = x_kinds.pop()
            else:  # mixed axes in one graph: keep the numbers as they are
                x_kind = "numeric"
            graphs.append({"name": graph.name, "kind": "graph",
                           "title": title or y_label or graph.name,
                           "x_label": x_label, "y_label": y_label, "x_kind": x_kind,
                           "traces": traces, "images": images,
                           "x_log": bool(_setting(x_axis, "log", False)) if x_axis else False,
                           "y_log": bool(_setting(y_axis, "log", False)) if y_axis else False,
                           "x_range": _axis_range(x_axis), "y_range": _axis_range(y_axis)})
        if graphs:
            pages.append({"name": page.name, "graphs": graphs})
    return pages


def _axis_range(axis: Any) -> Tuple[Optional[float], Optional[float]]:
    """(min, max) fixed in the project, None where the limit is Auto."""
    if axis is None:
        return None, None
    out = []
    for key in ("min", "max"):
        value = _setting(axis, key, "Auto")
        out.append(float(value) if isinstance(value, (int, float)) and
                   not isinstance(value, bool) else None)
    return out[0], out[1]


def _polar_graph(doc: Any, graph: Any) -> Optional[Dict[str, Any]]:
    """Description of a Veusz polar graph and its nonorthpoint traces."""
    traces = []
    for point in _children(graph, ("nonorthpoint",)):
        r_name, t_name = _setting(point, "data1"), _setting(point, "data2")
        r, _k1 = _numeric(doc, r_name)
        t, _k2 = _numeric(doc, t_name)
        if r is None or t is None:
            continue
        traces.append({"label": str(_setting(point, "key", "") or r_name),
                       "r_name": r_name, "t_name": t_name,
                       "colour": _setting(point, "PlotLine/color") or _setting(point, "MarkerFill/color")})
    if not traces:
        return None
    title_widget = next((c for c in _children(graph, ("label",)) if c.name == "plotTitle"), None)

    def number(key: str) -> Optional[float]:
        value = _setting(graph, key, "Auto")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    return {"name": graph.name, "kind": "polar", "traces": traces, "images": [],
            "title": str(_setting(title_widget, "label", "") or graph.name) if title_widget
            else graph.name,
            "rmin": number("minradius"), "rmax": number("maxradius"),
            "units": str(_setting(graph, "units", "degrees")),
            "direction": str(_setting(graph, "direction", "anticlockwise")),
            "position0": str(_setting(graph, "position0", "right")),
            "log": bool(_setting(graph, "log", False))}


def trace_arrays(doc: Any, trace: Dict[str, Any], x_kind: str,
                 mjd_as_date: bool) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    (x, y, x_is_unix_time) of one trace, converted for display.

    Veusz datetime seconds -> Unix seconds; MJD -> Unix seconds when
    ``mjd_as_date``; gaps broken with NaN.
    """
    x, _k = _numeric(doc, trace["x_name"])
    y, _k = _numeric(doc, trace["y_name"])
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    is_time = False
    if x_kind == "veusz_date":
        x = x + VEUSZ_EPOCH_UNIX
        is_time = True
    elif x_kind == "mjd" and mjd_as_date:
        x = (x - MJD_UNIX_EPOCH) * 86400.0
        is_time = True
    x, y = break_gaps(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return x, y, is_time


# %% One plot: all traces of one graph
class TracePlot(QtWidgets.QWidget):
    """
    A pyqtgraph plot with its legend-and-values column beside it
    (same layout and behaviour as the AB monitor's TracePlot).

    Signals
    -------
    hovered(float, float) : the mouse is over the data area at (x, value)
    unhovered()           : the mouse left the data area
    """
    hovered = Signal(float, float)
    unhovered = Signal()

    def __init__(self, title: str, x_label: str, y_label: str, date_axis: bool,
                 dark: bool = False, parent: Any = None) -> None:
        super().__init__(parent)
        self.title, self.date_axis = title, bool(date_axis)
        self.x_log = False                 # view x is log10(x) when True
        self.curves: Dict[str, Any] = {}
        self.rows: Dict[str, int] = {}
        self.data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal, self)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        bottom = (_DateAxisItem(orientation="bottom", utcOffset=0) if self.date_axis
                  else _AxisItem(orientation="bottom"))
        self.widget = _PlotWidget(viewBox=_ViewBox(),
                                  axisItems={"bottom": bottom,
                                             "left": _AxisItem(orientation="left")})
        self.widget.setMinimumHeight(PLOT_HEIGHT)
        self.plot = self.widget.getPlotItem()
        self.view = self.plot.getViewBox()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("left", y_label)
        self.plot.setLabel("bottom", "%s (UTC)" % x_label if self.date_axis and x_label
                           else ("UTC" if self.date_axis else x_label))
        self.view.setAutoVisible(y=True)
        self.vline = pg.InfiniteLine(angle=90, movable=False)
        self.hline = pg.InfiniteLine(angle=0, movable=False)
        for line in (self.vline, self.hline):
            self.plot.addItem(line, ignoreBounds=True)
            line.hide()
        self._proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=30,
                                     slot=self._mouse_moved)
        self.widget.left.connect(self.unhovered.emit)
        splitter.addWidget(self.widget)
        # --- legend column: [x] trace | at cursor | last ------------------------------
        side = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(side)
        column.setContentsMargins(4, 2, 2, 2)
        column.setSpacing(2)
        self.header = QtWidgets.QLabel()
        self.header.setTextFormat(Qt.TextFormat.RichText)
        self.header.setWordWrap(True)
        column.addWidget(self.header)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Trace", "At cursor", "Last"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.verticalHeader().setDefaultSectionSize(20)
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for col in (1, 2):
            head.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemChanged.connect(self._legend_toggled)
        column.addWidget(self.table, 1)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([900, LEGEND_WIDTH])
        self.apply_theme(dark)
        self._set_header(None)

    # ----- data -------------------------------------------------------------------
    def add_trace(self, label: str, x: np.ndarray, y: np.ndarray, colour: Optional[str] = None,
                  style: str = "solid", marker: str = "none", line: bool = True) -> None:
        """Add one trace (project colour, line style and marker when given)."""
        colour = colour if isinstance(colour, str) and colour.startswith("#") \
            else COLOURS[len(self.curves) % len(COLOURS)]
        n_finite = int(np.isfinite(y).sum())
        symbol = _MARKERS.get(str(marker))
        if symbol is None and (not line or n_finite <= SYMBOL_MAX_POINTS):
            symbol = "o"
        if n_finite > SYMBOL_MAX_POINTS and line:
            symbol = None                     # dense trace: the line carries it
        pen = pg.mkPen(colour, width=1.5,
                       style=_LINE_STYLES.get(str(style), Qt.PenStyle.SolidLine)) if line else None
        key = label
        suffix = 2
        while key in self.curves:             # keep duplicate legend names apart
            key = "%s (%d)" % (label, suffix)
            suffix += 1
        curve = self.plot.plot(x, y, pen=pen, symbol=symbol, symbolSize=4,
                               symbolBrush=colour, symbolPen=None, connect="finite")
        curve.setDownsampling(auto=True, method="peak")   # thin to the screen, keep spikes
        curve.setClipToView(True)
        self.curves[key] = curve
        self.data[key] = (x, y)
        row = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(row)
        item = QtWidgets.QTableWidgetItem(_swatch(colour), key)
        item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        item.setCheckState(Qt.CheckState.Checked)
        item.setToolTip("Untick to hide this trace")
        self.table.setItem(row, 0, item)
        for col in (1, 2):
            cell = QtWidgets.QTableWidgetItem("-")
            _align_right(cell)
            self.table.setItem(row, col, cell)
        ok = np.isfinite(y)
        self.table.item(row, 2).setText(_value_text(y[ok][-1]) if ok.any() else "-")
        self.table.blockSignals(False)
        self.rows[key] = row

    def apply_axes(self, x_log: bool = False, y_log: bool = False,
                   x_range: Tuple[Optional[float], Optional[float]] = (None, None),
                   y_range: Tuple[Optional[float], Optional[float]] = (None, None)) -> None:
        """Log axes and fixed limits from the project (None = Auto)."""
        if x_log or y_log:
            self.plot.setLogMode(x=bool(x_log), y=bool(y_log))
        self.x_log = bool(x_log) and not self.date_axis
        for (low, high), setter, log in ((x_range, self.plot.setXRange, x_log),
                                         (y_range, self.plot.setYRange, y_log)):
            if low is not None and high is not None and high > low:
                if log and low > 0:
                    low, high = np.log10(low), np.log10(high)
                elif log:
                    continue
                setter(low, high, padding=0.0)

    def set_region(self, start: float, stop: float, movable: bool = False,
                   on_change: Optional[Any] = None) -> None:
        """Shaded x region (e.g. a time gate); ``on_change(start, stop)`` when
        the user drags it (movable=True)."""
        region = getattr(self, "region", None)
        if region is None:
            region = pg.LinearRegionItem(values=(start, stop), movable=movable,
                                         brush=pg.mkBrush(255, 165, 0, 50))
            self.plot.addItem(region, ignoreBounds=True)
            self.region = region
            if on_change is not None:
                region.sigRegionChangeFinished.connect(
                    lambda item: on_change(*item.getRegion()))
        region.blockSignals(True)
        region.setRegion((start, stop))
        region.setMovable(movable)
        region.blockSignals(False)

    def dispose(self) -> None:
        """Free pyqtgraph's parentless right-click menus before deletion
        (they outlive deleted plots on PySide6 6.2; see ab_live_plots.py)."""
        try:
            self._proxy.disconnect()
        except Exception:
            pass
        for curve in self.curves.values():
            try:
                curve.clear()
            except Exception:
                pass
        self.data.clear()
        menus: List[Any] = [getattr(self.plot, "ctrlMenu", None)]
        menus.extend(getattr(self.plot, "subMenus", None) or [])
        view_menu = getattr(self.view, "menu", None)
        if view_menu is not None:
            menus.extend(getattr(view_menu, "axes", None) or [])
            menus.append(getattr(view_menu, "leftMenu", None))
            menus.append(view_menu)
        for menu in menus:
            if isinstance(menu, QtWidgets.QMenu):
                try:
                    menu.deleteLater()
                except RuntimeError:
                    pass

    # ----- crosshair and values -------------------------------------------------
    def value_at(self, label: str, x: float) -> Tuple[Optional[float], Optional[float]]:
        """(x, value) of trace ``label`` nearest to ``x``; (None, None) in a gap."""
        xs, ys = self.data.get(label, (np.empty(0), np.empty(0)))
        ok = np.isfinite(ys) & np.isfinite(xs)
        if not ok.any():
            return None, None
        xs, ys = xs[ok], ys[ok]
        order = None
        if np.any(np.diff(xs) < 0):            # unsorted x: search a sorted view
            order = np.argsort(xs, kind="mergesort")
            xs, ys = xs[order], ys[order]
        index = int(np.clip(np.searchsorted(xs, x), 0, len(xs) - 1))
        if index > 0 and abs(xs[index - 1] - x) <= abs(xs[index] - x):
            index -= 1
        if len(xs) > 1:
            steps = np.diff(xs)
            positive = steps[steps > 0]
            if positive.size and abs(xs[index] - x) > 1.5 * GAP_FACTOR * float(np.median(positive)):
                return None, None
        return float(xs[index]), float(ys[index])

    def set_cursor(self, x: Optional[float], y: Optional[float] = None) -> Dict[str, Optional[float]]:
        """Crosshair at ``x`` (and ``y`` on the hovered plot); fill the column."""
        values: Dict[str, Optional[float]] = {}
        self.table.blockSignals(True)
        if x is None:
            self.vline.hide()
            self.hline.hide()
            for row in self.rows.values():
                self.table.item(row, 1).setText("")
            self.table.blockSignals(False)
            self._set_header(None)
            return values
        self.vline.setPos(x)
        self.vline.show()
        if y is None:
            self.hline.hide()
        else:
            self.hline.setPos(y)
            self.hline.show()
        data_x = 10.0 ** x if self.x_log else x   # view x -> data x on a log axis
        for label, row in self.rows.items():
            _t, value = self.value_at(label, data_x)
            values[label] = value
            self.table.item(row, 1).setText(_value_text(value))
        self.table.blockSignals(False)
        self._set_header(data_x)
        return values

    def _set_header(self, x: Optional[float]) -> None:
        if x is None:
            where = "cursor -"
        elif self.date_axis:
            where = "cursor <b>%s</b>" % _utc_text(x)
        else:
            where = "cursor <b>%s</b>" % _value_text(x)
        self.header.setText("<b>%s</b><br>%s &nbsp;|&nbsp; %d trace(s)" % (
            self.title, where, len(self.curves)))

    def _mouse_moved(self, event: Any) -> None:
        pos = event[0]
        if not self.view.sceneBoundingRect().contains(pos):
            self.unhovered.emit()
            return
        point = self.view.mapSceneToView(pos)
        self.hovered.emit(float(point.x()), float(point.y()))

    def _legend_toggled(self, item: Any) -> None:
        if item.column() != 0:
            return
        curve = self.curves.get(item.text())
        if curve is not None:
            curve.setVisible(item.checkState() == Qt.CheckState.Checked)

    # ----- look -----------------------------------------------------------------
    def apply_theme(self, dark: bool) -> None:
        background, foreground, cross = THEMES["dark" if dark else "light"]
        self.widget.setBackground(background)
        for name in ("left", "bottom"):
            axis = self.plot.getAxis(name)
            axis.setPen(foreground)
            axis.setTextPen(foreground)
        pen = pg.mkPen(cross, style=Qt.PenStyle.DashLine)
        self.vline.setPen(pen)
        self.hline.setPen(pen)


class ImagePlot(QtWidgets.QWidget):
    """A 2-D dataset (FITS image HDU) with a colour bar."""

    def __init__(self, title: str, image: np.ndarray, dark: bool = False,
                 parent: Any = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("<b>%s</b> &nbsp; %d x %d" % (
            title, image.shape[1], image.shape[0])))
        self.widget = pg.PlotWidget()
        self.widget.setMinimumHeight(2 * PLOT_HEIGHT)
        self.widget.setAspectLocked(True)
        self.item = pg.ImageItem(image.T)          # pyqtgraph is column-major (x, y)
        self.widget.addItem(self.item)
        finite = image[np.isfinite(image)]
        levels = (float(np.percentile(finite, 1)), float(np.percentile(finite, 99))) \
            if finite.size else (0.0, 1.0)
        try:
            self.bar = self.widget.getPlotItem().addColorBar(
                self.item, colorMap="viridis", values=levels)
        except Exception:  # pyqtgraph < 0.12.3: no addColorBar
            self.item.setLevels(levels)
            self.bar = None
        layout.addWidget(self.widget)
        self.apply_theme(dark)

    def dispose(self) -> None:
        self.item.clear()

    def set_cursor(self, *_args: Any) -> Dict[str, Optional[float]]:
        return {}

    def apply_theme(self, dark: bool) -> None:
        self.widget.setBackground(THEMES["dark" if dark else "light"][0])


def _nice_ticks(low: float, high: float, count: int = 5) -> np.ndarray:
    """About ``count`` round tick values from low to high."""
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.array([low])
    raw = (high - low) / max(count, 1)
    magnitude = 10.0 ** np.floor(np.log10(raw))
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw),
               default=10 * magnitude)
    return np.arange(np.ceil(low / step) * step, high + step * 1e-9, step)


class _GridPlot(QtWidgets.QWidget):
    """Common base of PolarPlot and SmithPlot: an aspect-locked plot with a
    drawn grid, and the same legend-and-values column as TracePlot."""
    hovered = Signal(float, float)
    unhovered = Signal()
    date_axis = False

    def __init__(self, title: str, dark: bool = False, parent: Any = None) -> None:
        super().__init__(parent)
        self.title = title
        self.curves: Dict[str, Any] = {}
        self.rows: Dict[str, int] = {}
        self.data: Dict[str, Any] = {}
        self.grid_items: List[Any] = []
        self.grid_labels: List[Any] = []
        splitter = QtWidgets.QSplitter(Qt.Orientation.Horizontal, self)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.widget = _PlotWidget(viewBox=_ViewBox())
        self.widget.setMinimumHeight(int(1.6 * PLOT_HEIGHT))
        self.plot = self.widget.getPlotItem()
        self.view = self.plot.getViewBox()
        self.view.setAspectLocked(True)
        self.plot.hideAxis("left")
        self.plot.hideAxis("bottom")
        self.marker = pg.ScatterPlotItem(size=9, pen=pg.mkPen("w"), brush=pg.mkBrush(None))
        self.plot.addItem(self.marker, ignoreBounds=True)
        self._proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=30,
                                     slot=self._mouse_moved)
        self.widget.left.connect(self.unhovered.emit)
        splitter.addWidget(self.widget)
        side = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(side)
        column.setContentsMargins(4, 2, 2, 2)
        self.header = QtWidgets.QLabel()
        self.header.setTextFormat(Qt.TextFormat.RichText)
        self.header.setWordWrap(True)
        column.addWidget(self.header)
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Trace", "At cursor"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemChanged.connect(self._legend_toggled)
        column.addWidget(self.table, 1)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([900, LEGEND_WIDTH])
        self._dark = bool(dark)

    def _grid_line(self, x: np.ndarray, y: np.ndarray, major: bool = False) -> None:
        item = pg.PlotCurveItem(x, y, connect="finite")
        item.setZValue(-10)
        item._major = major
        self.plot.addItem(item, ignoreBounds=False)
        self.grid_items.append(item)

    def _grid_text(self, text: str, x: float, y: float, anchor=(0.5, 0.5)) -> None:
        label = pg.TextItem(text, anchor=anchor)
        label.setPos(x, y)
        self.plot.addItem(label, ignoreBounds=True)
        self.grid_labels.append(label)

    def _add_curve(self, label: str, x: np.ndarray, y: np.ndarray, colour: Optional[str]) -> str:
        colour = colour if isinstance(colour, str) and colour.startswith("#") \
            else COLOURS[len(self.curves) % len(COLOURS)]
        key, suffix = label, 2
        while key in self.curves:
            key = "%s (%d)" % (label, suffix)
            suffix += 1
        curve = self.plot.plot(x, y, pen=pg.mkPen(colour, width=1.8), connect="finite")
        self.curves[key] = curve
        row = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(row)
        item = QtWidgets.QTableWidgetItem(_swatch(colour), key)
        item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        item.setCheckState(Qt.CheckState.Checked)
        self.table.setItem(row, 0, item)
        cell = QtWidgets.QTableWidgetItem("-")
        _align_right(cell)
        self.table.setItem(row, 1, cell)
        self.table.blockSignals(False)
        self.rows[key] = row
        return key

    def _legend_toggled(self, item: Any) -> None:
        if item.column() == 0 and item.text() in self.curves:
            self.curves[item.text()].setVisible(item.checkState() == Qt.CheckState.Checked)

    def _mouse_moved(self, event: Any) -> None:
        pos = event[0]
        if not self.view.sceneBoundingRect().contains(pos):
            self.unhovered.emit()
            return
        point = self.view.mapSceneToView(pos)
        self.hovered.emit(float(point.x()), float(point.y()))

    def dispose(self) -> None:
        try:
            self._proxy.disconnect()
        except Exception:
            pass
        for curve in self.curves.values():
            curve.clear()
        self.data.clear()

    def apply_theme(self, dark: bool) -> None:
        self._dark = bool(dark)
        background, foreground, cross = THEMES["dark" if dark else "light"]
        self.widget.setBackground(background)
        for item in self.grid_items:
            item.setPen(pg.mkPen(foreground if item._major else cross,
                                 width=1.2 if item._major else 0.6,
                                 style=Qt.PenStyle.SolidLine if item._major
                                 else Qt.PenStyle.DotLine))
        for label in self.grid_labels:
            label.setColor(foreground)
        self.marker.setPen(pg.mkPen(foreground, width=1.5))


class PolarPlot(_GridPlot):
    """Veusz-style polar plot of r(theta) traces (e.g. antenna patterns).

    The parameters are those of the Veusz polar widget [P5]:
    ``rmin``/``rmax`` (None = data range), ``units`` "degrees", "radians",
    "fractions" or "percentages", ``direction`` "clockwise" or
    "anticlockwise", ``position0`` "right", "top", "left" or "bottom", and
    ``log`` for a logarithmic radial axis.  As in Veusz, radii below rmin
    are drawn at the centre.
    """

    _ZERO = {"right": 0.0, "top": 0.5 * np.pi, "left": np.pi, "bottom": 1.5 * np.pi}
    _TO_RADIANS = {"degrees": np.pi / 180.0, "radians": 1.0,
                   "fractions": 2.0 * np.pi, "percentages": 0.02 * np.pi}
    _PERIOD = {"degrees": 360.0, "radians": 2.0 * np.pi, "fractions": 1.0,
               "percentages": 100.0}

    def __init__(self, title: str, rmin: Optional[float], rmax: Optional[float],
                 units: str = "degrees", direction: str = "anticlockwise",
                 position0: str = "right", log: bool = False, dark: bool = False,
                 parent: Any = None) -> None:
        super().__init__(title, dark, parent)
        self.rmin, self.rmax = rmin, rmax
        self.units = units if units in self._TO_RADIANS else "degrees"
        self.sign = -1.0 if direction == "clockwise" else 1.0
        self.zero = self._ZERO.get(position0, 0.0)
        self.log = bool(log)
        self._raw: List[Tuple[str, np.ndarray, np.ndarray, Optional[str]]] = []
        self.header.setText("<b>%s</b>" % html_escape(title))

    def _screen_angle(self, theta: np.ndarray) -> np.ndarray:
        """Data angle (project units) -> mathematical angle on screen."""
        return self.zero + self.sign * np.asarray(theta, dtype=float) * self._TO_RADIANS[self.units]

    def _data_angle(self, phi: float) -> float:
        """Inverse of _screen_angle, wrapped to (-period/2, period/2]."""
        period = self._PERIOD[self.units]
        value = (phi - self.zero) * self.sign / self._TO_RADIANS[self.units]
        return float((value + period / 2.0) % period - period / 2.0)

    def _scale(self, r: np.ndarray) -> np.ndarray:
        """Radius -> 0..1 (Veusz toPlotRadius)."""
        r = np.asarray(r, dtype=float)
        if self.log:
            low, high = np.log(self.rmin), np.log(self.rmax)
            scaled = (np.log(np.clip(r, 1e-99, 1e99)) - low) / (high - low)
        else:
            scaled = (r - self.rmin) / (self.rmax - self.rmin)
        return np.where(scaled > 0.0, scaled, 0.0)

    def _unscale(self, rho: float) -> float:
        if self.log:
            return float(np.exp(np.log(self.rmin) + rho * (np.log(self.rmax) - np.log(self.rmin))))
        return float(self.rmin + rho * (self.rmax - self.rmin))

    def add_trace(self, label: str, r: np.ndarray, theta: np.ndarray,
                  colour: Optional[str] = None) -> None:
        """Queue one trace; finish() scales and draws all of them."""
        if r is None or theta is None:
            return
        n = min(len(r), len(theta))
        self._raw.append((label, np.asarray(r[:n], dtype=float),
                          np.asarray(theta[:n], dtype=float), colour))

    def finish(self) -> None:
        """Fix the radial range (project or data), draw grid and traces."""
        values = [r[np.isfinite(r) & ((r > 0) if self.log else True)]
                  for _l, r, _t, _c in self._raw]
        values = np.concatenate(values) if values else np.empty(0)
        if self.rmin is None:
            self.rmin = float(np.min(values)) if values.size else (0.1 if self.log else 0.0)
        if self.rmax is None:
            self.rmax = float(np.max(values)) if values.size else 1.0
        if self.log and self.rmin <= 0:
            self.rmin = self.rmax * 1e-3 if self.rmax > 0 else 1e-3
        if self.rmax <= self.rmin:
            self.rmax = self.rmin * 10.0 if self.log else self.rmin + 1.0
        self._draw_grid()
        for label, r, theta, colour in self._raw:
            phi = self._screen_angle(theta)
            rho = self._scale(r)
            key = self._add_curve(label, rho * np.cos(phi), rho * np.sin(phi), colour)
            self.data[key] = (r, theta)
        self._raw = []
        self.apply_theme(self._dark)
        self.view.setRange(xRange=(-1.2, 1.2), yRange=(-1.2, 1.2), padding=0.0)

    def _draw_grid(self) -> None:
        circle = np.linspace(0.0, 2.0 * np.pi, 361)
        if self.log:
            ticks = 10.0 ** np.arange(np.ceil(np.log10(self.rmin)), np.floor(np.log10(self.rmax)) + 1)
        else:
            ticks = _nice_ticks(self.rmin, self.rmax, 5)
        label_angle = self.zero + 0.5 * self.sign * np.pi / 12.0
        for value in ticks:
            rho = float(self._scale(np.array([value]))[0])
            if 0.0 < rho <= 1.0001:
                self._grid_line(rho * np.cos(circle), rho * np.sin(circle), major=False)
                self._grid_text("%g" % value, rho * np.cos(label_angle),
                                rho * np.sin(label_angle), anchor=(0.5, 0.5))
        self._grid_line(np.cos(circle), np.sin(circle), major=True)
        period = self._PERIOD[self.units]
        for step in range(12):                            # spokes every 30 degrees
            theta = step * period / 12.0
            phi = float(self._screen_angle(np.array([theta]))[0])
            self._grid_line(np.array([0.0, np.cos(phi)]), np.array([0.0, np.sin(phi)]))
            shown = theta if theta <= period / 2.0 else theta - period
            text = ("%g°" % shown) if self.units == "degrees" else "%.4g" % shown
            self._grid_text(text, 1.1 * np.cos(phi), 1.1 * np.sin(phi))

    def set_cursor(self, x: Optional[float], y: Optional[float] = None) -> Dict[str, Optional[float]]:
        """Angle under the mouse; each trace's r at the nearest angle."""
        values: Dict[str, Optional[float]] = {}
        self.table.blockSignals(True)
        if x is None or y is None:
            for row in self.rows.values():
                self.table.item(row, 1).setText("")
            self.marker.clear()
            self.header.setText("<b>%s</b>" % html_escape(self.title))
            self.table.blockSignals(False)
            return values
        theta_here = self._data_angle(float(np.arctan2(y, x)))
        period = self._PERIOD[self.units]
        spots = []
        for label, row in self.rows.items():
            r, t = self.data.get(label, (np.empty(0), np.empty(0)))
            ok = np.isfinite(r) & np.isfinite(t)
            if not ok.any():
                values[label] = None
                continue
            diff = np.abs((t[ok] - theta_here + period / 2.0) % period - period / 2.0)
            index = int(np.argmin(diff))
            values[label] = float(r[ok][index])
            self.table.item(row, 1).setText(_value_text(values[label]))
            angle = float(self._screen_angle(np.array([t[ok][index]]))[0])
            scaled = float(self._scale(np.array([r[ok][index]]))[0])
            spots.append({"pos": (scaled * np.cos(angle), scaled * np.sin(angle))})
        self.marker.setData(spots)
        unit = {"degrees": "deg", "radians": "rad", "fractions": "turn",
                "percentages": "%"}[self.units]
        self.header.setText("<b>%s</b><br>cursor %s %s, r %s" % (
            html_escape(self.title), _value_text(theta_here), unit,
            _value_text(self._unscale(min(float(np.hypot(x, y)), 1.0)))))
        self.table.blockSignals(False)
        return values


class SmithPlot(_GridPlot):
    """Smith chart (impedance grid "z" or admittance grid "y") of reflection
    coefficients, with f, Gamma, Z and VSWR of the nearest point under the
    mouse.  The data are always Gamma; the "y" grid is the "z" grid rotated
    by 180 degrees, as scikit-rf draws it."""

    R_VALUES = (0.2, 0.5, 1.0, 2.0, 5.0)
    X_VALUES = (0.2, 0.5, 1.0, 2.0, 5.0)

    def __init__(self, title: str, chart_type: str = "z", z0: float = 50.0,
                 dark: bool = False, parent: Any = None) -> None:
        super().__init__(title, dark, parent)
        self.chart_type = chart_type
        self.z0 = float(z0)
        self.freqs: Dict[str, np.ndarray] = {}
        self._draw_grid()
        self.header.setText("<b>%s</b>" % html_escape(title))
        self.apply_theme(dark)

    def _draw_grid(self) -> None:
        mirror = -1.0 if self.chart_type == "y" else 1.0
        circle = np.linspace(0.0, 2.0 * np.pi, 721)
        self._grid_line(np.cos(circle), np.sin(circle), major=True)
        self._grid_line(np.array([-1.0, 1.0]), np.array([0.0, 0.0]))
        for r in self.R_VALUES:                       # constant resistance circles
            centre, radius = r / (1.0 + r), 1.0 / (1.0 + r)
            self._grid_line(mirror * (centre + radius * np.cos(circle)), radius * np.sin(circle))
            self._grid_text("%g" % r, mirror * (centre - radius), 0.0, anchor=(0, 0))
        t = np.linspace(-np.pi, np.pi, 1441)
        for x in self.X_VALUES:                       # constant reactance arcs
            for sign in (1.0, -1.0):
                xs = sign * x
                gx, gy = 1.0 + np.cos(t) / x, 1.0 / xs + np.sin(t) / x
                inside = gx ** 2 + gy ** 2 <= 1.0 + 1e-9
                gx = np.where(inside, gx, np.nan)
                gy = np.where(inside, gy, np.nan)
                self._grid_line(mirror * gx, mirror * gy)
                angle = 2.0 * np.arctan(1.0 / xs)       # where the arc meets |Gamma| = 1
                self._grid_text("%+gj" % xs, 1.07 * mirror * np.cos(angle),
                                1.07 * mirror * np.sin(angle))

    def add_trace(self, label: str, gamma: np.ndarray, freq_hz: Optional[np.ndarray] = None,
                  colour: Optional[str] = None) -> None:
        gamma = np.asarray(gamma, dtype=complex)
        key = self._add_curve(label, gamma.real, gamma.imag, colour)
        self.data[key] = gamma
        self.freqs[key] = np.asarray(freq_hz, dtype=float) if freq_hz is not None \
            else np.arange(gamma.size, dtype=float)
        self.apply_theme(self._dark)
        self.view.setRange(xRange=(-1.15, 1.15), yRange=(-1.15, 1.15), padding=0.0)

    def set_cursor(self, x: Optional[float], y: Optional[float] = None) -> Dict[str, Optional[float]]:
        values: Dict[str, Optional[float]] = {}
        self.table.blockSignals(True)
        if x is None or y is None:
            for row in self.rows.values():
                self.table.item(row, 1).setText("")
            self.marker.clear()
            self.header.setText("<b>%s</b>" % self.title)
            self.table.blockSignals(False)
            return values
        here = complex(x, y)
        best = None
        for label, row in self.rows.items():
            gamma = self.data.get(label)
            if gamma is None or gamma.size == 0:
                continue
            index = int(np.nanargmin(np.abs(gamma - here)))
            values[label] = float(self.freqs[label][index])
            self.table.item(row, 1).setText("%.6g GHz" % (self.freqs[label][index] / 1e9))
            distance = abs(gamma[index] - here)
            if best is None or distance < best[0]:
                best = (distance, label, index)
        if best is not None:
            _d, label, index = best
            g = self.data[label][index]
            z = self.z0 * (1 + g) / (1 - g) if g != 1 else complex(np.inf, 0)
            mag = abs(g)
            vswr = (1 + mag) / (1 - mag) if mag < 1 else np.inf
            self.marker.setData([{"pos": (g.real, g.imag)}])
            self.header.setText(
                "<b>%s</b><br>%s @ %.6g GHz<br>|&Gamma;| %.4f &ang; %.2f&deg; "
                "(%.2f dB)<br>Z = %.4g %+.4gj &Omega; &nbsp; VSWR %.3f" % (
                    self.title, html_escape(label), self.freqs[label][index] / 1e9, mag,
                    np.degrees(np.angle(g)), 20 * np.log10(max(mag, 1e-12)),
                    z.real, z.imag, vswr))
        self.table.blockSignals(False)
        return values


def html_escape(text: str) -> str:
    """Minimal HTML escaping for header labels."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# %% The panel: page navigation and a scrolling column of plots
class ProjectPlotPanel(QtWidgets.QWidget):
    """
    The "Plots" tab.  Call ``set_document(doc)`` after processing; only the
    selected page has plot widgets.
    """

    def __init__(self, dark: bool = False, parent: Any = None) -> None:
        super().__init__(parent)
        self.doc: Any = None
        self.pages: List[Dict[str, Any]] = []
        self.plots: List[Any] = []
        self.dark = bool(dark)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Page:"))
        self.page_box = QtWidgets.QComboBox()
        self.page_box.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.page_box.setMinimumContentsLength(40)
        self.page_box.currentIndexChanged.connect(self._page_changed)
        bar.addWidget(self.page_box, 1)
        self.prev_button = QtWidgets.QPushButton("<")
        self.prev_button.setToolTip("Previous page")
        self.prev_button.clicked.connect(lambda: self._step(-1))
        self.next_button = QtWidgets.QPushButton(">")
        self.next_button.setToolTip("Next page")
        self.next_button.clicked.connect(lambda: self._step(1))
        for button in (self.prev_button, self.next_button):
            button.setFixedWidth(32)
            bar.addWidget(button)
        self.mjd_box = QtWidgets.QCheckBox("MJD as UTC date")
        self.mjd_box.setChecked(True)
        self.mjd_box.setToolTip("Draw MJD x axes as UTC dates (values are unchanged).")
        self.mjd_box.toggled.connect(lambda _on: self._page_changed(self.page_box.currentIndex()))
        bar.addWidget(self.mjd_box)
        fit = QtWidgets.QPushButton("Fit all")
        fit.setToolTip("Auto-range every plot of this page")
        fit.clicked.connect(self.fit_all)
        bar.addWidget(fit)
        hint = QtWidgets.QLabel("drag: pan · right-drag: zoom axis · "
                                + ("Ctrl+wheel" if WHEEL_NEEDS_CTRL else "wheel")
                                + ": zoom · right-click: menu / export")
        hint.setStyleSheet("color: #8c8fa1;")
        bar.addWidget(hint)
        outer.addLayout(bar)
        self.page = QtWidgets.QWidget()
        self.column = QtWidgets.QVBoxLayout(self.page)
        self.column.setContentsMargins(0, 0, 0, 0)
        self.column.setSpacing(6)
        self.empty = QtWidgets.QLabel("Process files to see the plots here.")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.column.addWidget(self.empty)
        self.column.addStretch(1)
        self.scroller = QtWidgets.QScrollArea()
        self.scroller.setWidgetResizable(True)
        self.scroller.setWidget(self.page)
        outer.addWidget(self.scroller, 1)
        self._update_buttons()

    # ----- document --------------------------------------------------------------
    def set_document(self, doc: Any) -> int:
        """Read the pages of ``doc`` and show the first; returns the page count."""
        current = self.page_box.currentData()
        self.doc = doc
        self.pages = document_pages(doc) if doc is not None else []
        self.page_box.blockSignals(True)
        self.page_box.clear()
        for page in self.pages:
            traces = sum(len(g["traces"]) + len(g["images"]) for g in page["graphs"])
            self.page_box.addItem("%s  (%d graphs, %d traces)" % (
                page["name"], len(page["graphs"]), traces), page["name"])
        names = [p["name"] for p in self.pages]
        index = names.index(current) if current in names else 0
        self.page_box.setCurrentIndex(index if self.pages else -1)
        self.page_box.blockSignals(False)
        self._page_changed(self.page_box.currentIndex())
        return len(self.pages)

    def clear(self) -> None:
        """Forget the document and free every plot."""
        self.set_document(None)

    # ----- page ------------------------------------------------------------------
    def _step(self, delta: int) -> None:
        index = self.page_box.currentIndex() + delta
        if 0 <= index < self.page_box.count():
            self.page_box.setCurrentIndex(index)

    def _update_buttons(self) -> None:
        index, count = self.page_box.currentIndex(), self.page_box.count()
        self.prev_button.setEnabled(index > 0)
        self.next_button.setEnabled(0 <= index < count - 1)

    def _clear_plots(self) -> None:
        for plot in self.plots:
            plot.dispose()
            self.column.removeWidget(plot)
            plot.setParent(None)
            plot.deleteLater()
        self.plots = []

    def _page_changed(self, index: int) -> None:
        """Build the plots of page ``index`` (the previous page's are freed)."""
        self._clear_plots()
        self._update_buttons()
        if self.doc is None or not (0 <= index < len(self.pages)):
            self.empty.setText("Process files to see the plots here." if self.doc is None
                               else "This project has no plottable pages.")
            self.empty.setVisible(True)
            return
        self.empty.setVisible(False)
        page = self.pages[index]
        master = None
        mjd_as_date = self.mjd_box.isChecked()
        for position, graph in enumerate(page["graphs"]):
            if graph.get("kind") == "polar":
                plot = PolarPlot(graph["title"], graph["rmin"], graph["rmax"],
                                 graph["units"], graph["direction"], graph["position0"],
                                 graph.get("log", False), dark=self.dark)
                for trace in graph["traces"]:
                    r, _k1 = _numeric(self.doc, trace["r_name"])
                    t, _k2 = _numeric(self.doc, trace["t_name"])
                    plot.add_trace(trace["label"], r, t, trace["colour"])
                plot.finish()
                plot.hovered.connect(self._hovered)
                plot.unhovered.connect(self._unhovered)
                self.column.insertWidget(len(self.plots), plot)
                self.plots.append(plot)
                continue
            if graph["traces"]:
                date_axis = graph["x_kind"] == "veusz_date" or (
                    graph["x_kind"] == "mjd" and mjd_as_date)
                plot = TracePlot(graph["title"], graph["x_label"], graph["y_label"],
                                 date_axis, dark=self.dark)
                for trace in graph["traces"]:
                    x, y, _is_time = trace_arrays(self.doc, trace, graph["x_kind"],
                                                  mjd_as_date)
                    plot.add_trace(trace["label"], x, y, trace["colour"], trace["style"],
                                   trace["marker"], trace["line"])
                plot._set_header(None)
                plot.apply_axes(graph.get("x_log", False), graph.get("y_log", False),
                                graph.get("x_range", (None, None)) if not date_axis
                                else (None, None),
                                graph.get("y_range", (None, None)))
                plot.hovered.connect(self._hovered)
                plot.unhovered.connect(self._unhovered)
                if master is None:
                    master = plot
                elif master.date_axis == plot.date_axis:
                    plot.plot.setXLink(master.plot)    # one page: zoom in x together
                self.column.insertWidget(len(self.plots), plot)
                self.plots.append(plot)
            for name in graph["images"]:
                data = np.asarray(self.doc.datasets[name][1]["data"], dtype=float)
                plot = ImagePlot("%s - %s" % (graph["title"], name), data, dark=self.dark)
                self.column.insertWidget(len(self.plots), plot)
                self.plots.append(plot)
        self.scroller.verticalScrollBar().setValue(0)

    def fit_all(self) -> None:
        for plot in self.plots:
            if isinstance(plot, TracePlot):
                plot.view.enableAutoRange()
            elif isinstance(plot, _GridPlot):
                plot.view.setRange(xRange=(-1.2, 1.2), yRange=(-1.2, 1.2), padding=0.0)

    # ----- crosshair -----------------------------------------------------------
    def _hovered(self, x: float, y: float) -> None:
        source = self.sender()
        if isinstance(source, _GridPlot):          # polar / Smith: its own cursor only
            source.set_cursor(x, y)
            return
        for plot in self.plots:
            if isinstance(plot, TracePlot) and plot.date_axis == getattr(source, "date_axis", None):
                plot.set_cursor(x, y if plot is source else None)

    def _unhovered(self) -> None:
        for plot in self.plots:
            plot.set_cursor(None)

    def apply_theme(self, dark: bool) -> None:
        self.dark = bool(dark)
        for plot in self.plots:
            plot.apply_theme(dark)


def self_test() -> Optional[str]:
    """Build and delete one plot; None when pyqtgraph works, else the reason
    (needs a QApplication).  pyqtgraph 0.12.4 imports with PySide6 >= 6.3 but
    fails as soon as a plot gets a label, so importing alone proves nothing."""
    plot = None
    try:
        plot = TracePlot("self-test", "x", "y", date_axis=True)
        plot.add_trace("a", np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        plot.set_cursor(1.5, 3.5)
        return None
    except Exception as exc:
        return "%s: %s" % (type(exc).__name__, exc)
    finally:
        if plot is not None:
            plot.dispose()
            plot.setParent(None)
            plot.deleteLater()
