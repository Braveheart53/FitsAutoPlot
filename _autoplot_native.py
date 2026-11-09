# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
_autoplot_native.py

Native AutoPlot project model: a small, veusz.embed-like document (datasets,
widget tree, settings) that the AutoPlot page builders fill and that is
written as a Veusz HDF5 project (.vszh5) by vszh5_writer.  No Qt or Veusz
imports.

The public SetData/Root.Add surface is deliberately small, so existing
AutoPlot page builders can target this model without a running Veusz process.
This is not a general-purpose replacement for the Veusz scripting API.

On save, finalize() names auto-named widgets after their data and lays out
every time axis: MJD/DMJD and Veusz date datasets become local ISO 8601
date-time axes, a trace's line breaks at a small gap (dt > K x median dt)
and the x axis breaks at a large gap (dt > n x the small-gap threshold) -
see TimeLayout.

Created on 2026-10-11

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.5.0: Time axes laid out on save: local ISO 8601, small/large gaps.
Date: 2026-11-09
#              * finalize() lays out every horizontal axis whose traces are
#                against time - MJD/DMJD (1941-2132; MJD 0 padding ignored),
#                a Veusz date dataset, or a seconds column: MJD becomes a
#                Veusz date-time plot copy "<x>_local" (DST-aware local
#                clock, ISO 8601 ticks with the UTC offset, label "Local
#                time (ISO 8601, UTC-05:00)"); date datasets get the same.
#              * TimeLayout(k, n, manual_s, axis_manual_s) on the document
#                (set by the GUIs): SMALL gap - a trace's step dt >
#                k x median(dt) of that trace (or manual_s) - the line breaks
#                (NaN rows in the plot copies "<x>_local" / "<y>_gaps"; the
#                stored data are unchanged); LARGE gap - no trace on the
#                axis has data for dt > n x the largest small-gap threshold
#                (or axis_manual_s) - the x axis becomes axis-broken there.
#                Defaults k = 10, n = 100.  Re-laid out when the settings
#                change; the rule is written in each graph's notes.
#              * Pages with date-time axes and no explicit size get room for
#                the rotated ISO tick labels (13 cm per grid column, 8 cm per
#                row); date axes use 8 pt ticks and a 10 pt label; a DST
#                label names each real offset change once.
#              * Writer 0.0.27.
#              * Header in the author's template; docstrings for every
#                function, class and method.
# %%%%% Function Descriptions
        _plain: Plain Python literal value of a setting (numpy -> Python); validated.
        _literal: Python literal text of a setting (kept for callers of 0.1.0).
        _escape_name: Veusz HDF5 dataset-name escaping (vszh5_writer.escape_hdf_name).
        _Setting (class): Proxy for one setting path of a NativeWidget (``widget.Label.size``).
        NativeWidget (class): One widget of the native project tree (page, graph, axis, xy, ...).
        mark_local_time_axis: Make ``axis`` (a NativeWidget axis or axis-broken) a local date-time axis (0.3.0): its tick format and label are set from the local datasets drawn...
        TimeLayout (class): Gap rule of the time axes (0.5.0), see vszh5_writer TIME GAPS.
        _seconds_text: A threshold for people: '45 s', '12 min', '3.5 h', '2.1 d'.
        _time_kind: 'date' (Veusz date dataset, UTC), 'mjd', 'seconds' or None (not time).
        _plot_name: A dataset name for a plot copy: ``wanted``, or wanted_2, ...
        _layout_time_axis: Lay out one x axis whose traces are against time (0.5.0).
        _size_dated_page: Give a page with date-time axes room for its ISO 8601 tick labels (0.5.0).
        apply_settings: Set ``{setting path: value}`` on a widget (0.4.0).
        add_smith_graph: Add a Smith chart graph (vszh5_writer.smith_graph_spec) under ``parent`` (a page or grid widget); returns the graph widget (0.4.0).
        add_smith_page: A page with one Smith chart, as vszh5_writer.smith_page() builds it, on a NativeDocument (or veusz.embed) widget tree; returns the graph (0.4.0).
        _data_name: The dataset a plotting widget draws (y for xy), or ''.
        _rename_from_data: Name an auto-named widget after its data (unique among siblings).
        NativeDocument (class): Build and atomically save a self-contained Veusz format-1 document.
# %%%%% Variable Descriptions
    Define all utilized variables
        VERSION: module version
        WRITER_VERSION: version of the vszh5_writer in use
        DATASET_SETTINGS: widget settings that name datasets (checked on save)
        WIDGET_KINDS: widget types the native model supports
        MJD_VEUSZ_EPOCH: MJD of the Veusz date origin 2009-01-01T00:00:00 UTC.
        MJD_RANGE: Plausible MJD range of a time column (1941 .. 2132); values outside are : not taken for MJD (and MJD 0 padding rows are ignored).
        _TIME_NAME_HINT: column / axis names that suggest time (MJD, DMJD, JD, TIME, DATE)
        _SECONDS_NAMES: column names whose values are seconds
        _DATA_SETTING_OF: the setting that holds each widget kind's main dataset (widget names)
# %%%%% More Info
    Imported by every AutoPlot program and plugin; needs numpy, h5py and
    vszh5_writer only (no Qt, no Veusz).
    Tests: python -m unittest discover -s tests
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.4.0: Smith charts in the project, writer 0.0.26.
Date: 2026-11-09
#              * add_smith_page(root, page, grid, traces, ...) and
#                add_smith_graph(parent, grid, traces, ...): a Veusz Smith
#                chart (vszh5_writer.smith_graph_spec: aspect-locked graph,
#                hidden Re/Im Gamma axes, grid circles and arcs as xy lines,
#                grid values as point labels, traces of Im against Re Gamma,
#                legend beside the chart) on a NativeDocument - or on a
#                veusz.embed document, through apply_settings().
#              * apply_settings(widget, {path: value}): one specification for
#                both widget trees.
#              * Writer 0.0.26.
# %%%% 0.3.0: Local ISO date-time axes, names from the data, writer 0.0.25.
Date: 2026-11-09
#              * SetDataLocalTime(utc_name): from a stored UTC date dataset,
#                a second date dataset "<name>_local" on the clock of the
#                computer doing the processing (VSZH5_DISPLAY_TZ overrides),
#                DST-aware (vszh5_writer.local_time_from_veusz).  The UTC
#                values stay the stored Veusz date-time; plots bind the local
#                one.
#              * mark_local_time_axis(axis): the axis is a datetime axis whose
#                tick format and label are resolved from the local datasets
#                drawn on it when the document is finalised: ISO 8601,
#                24-hour, with the UTC offset, e.g.
#                %VDY-%VDm-%VDdT%VDH:%VDM:%VDS-05:00 and the label "Local
#                time (ISO 8601, UTC-05:00)"; across a DST change the label
#                lists every offset (writer 0.0.24 rules).  Tick labels are
#                rotated 45 degrees.  Traces on a marked axis that bind a
#                UTC date dataset with a local copy draw the local copy.
#              * Widgets added without a name (Add("xy")) are named after
#                their data when the document is finalised: xy -> the y
#                dataset, nonorthpoint -> the r dataset, graph/polar -> "g_"
#                + the first trace's dataset (unique among siblings) instead
#                of "xy", "xy_1", "graph" ...  Named widgets are unchanged.
#              * finalize() runs both; save() and SaveToString() call it,
#                and the pyqtgraph Plots tab reads the finalised tree.
#              * Writer 0.0.25 (names, notes and local-time helpers).
# %%%% 0.2.0: Files written through vszh5_writer 0.0.24 (the shared master).
Date: 2026-11-05
#              The widget tree and dataset API are unchanged (SetData,
#              Root.Add, TagDatasets, save/Save ...).  Serialisation now goes
#              through the author's shared writer, vszh5_writer.py 0.0.24
#              (SHA-256 199f30a8...a901, master copy in the RF project):
#              VeuszWriter writes the datasets, tags and document atomically
#              (private ".part" file, os.replace), VszDoc builds the command
#              script and vz_text escapes captions (now also '[', ']' and a
#              backslash, Veusz's own latexEscape).  Large arrays are
#              gzip+shuffle compressed (>= 4096 elements).
#              New widget kinds for the ATR, CSV/TSV and R&S tools: polar,
#              nonorthpoint, function, bar.  New NativeWidget.typename and
#              childnames (veusz.embed names), data1/data2/lengths count as
#              dataset references (minimized saves), numpy scalars are stored
#              as plain Python values.
# %%%% 0.1.0: Direct .vszh5 serialization using h5py.
Date: 2026-10-11
=============================================================================
"""
from __future__ import annotations

import math
import os
import re
from collections import OrderedDict

import numpy as np

import vszh5_writer
from vszh5_writer import VeuszWriter, VszDoc, vz_text

VERSION = "0.5.0"
WRITER_VERSION = vszh5_writer.__version__
DATASET_SETTINGS = frozenset((
    "xData", "yData", "labels", "scalePoints", "yDataLow", "yDataHigh",
    "xDataLow", "xDataHigh", "data", "data1", "data2", "lengths",
))
WIDGET_KINDS = frozenset((
    "page", "grid", "graph", "axis", "axis-broken", "xy", "key", "image",
    "colorbar", "label", "polar", "nonorthpoint", "function", "bar",
))


# %% Safe script values
def _plain(value):
    """Plain Python literal value of a setting (numpy -> Python); validated."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite widget setting (dataset NaNs are supported)")
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError("Unsupported setting value: %s" % type(value).__name__)
    return value


def _literal(value):
    """Python literal text of a setting (kept for callers of 0.1.0)."""
    return repr(_plain(value))


def _escape_name(name):
    """Veusz HDF5 dataset-name escaping (vszh5_writer.escape_hdf_name)."""
    return vszh5_writer.escape_hdf_name(name)


class _Setting:
    """Proxy for one setting path of a NativeWidget (``widget.Label.size``).

    Mimics the veusz.embed setting objects: attribute access walks deeper into
    the setting path ("Label" -> "Label/size") and ``.val`` reads or writes the
    value stored in the widget's settings dictionary.
    """
    def __init__(self, widget, path):
        """Remember the owning widget and the '/'-separated setting path."""
        self._widget = widget
        self._path = path

    def __getattr__(self, name):
        """Return the proxy for the child setting ``<path>/<name>``.

        Names starting with '_' raise AttributeError so copy/pickle and hasattr
        probes do not create bogus setting paths.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return _Setting(self._widget, self._path + "/" + name)

    @property
    def val(self):
        """Value of the setting, or None when it has never been set."""
        return self._widget.settings.get(self._path)

    @val.setter
    def val(self, value):
        """Store ``value`` (converted to plain Python/numpy types first, so a bad
        value fails here and leaves the widget unchanged).
        """
        self._widget.settings[self._path] = _plain(value)   # fail before modifying


# %% In-memory widget tree
class NativeWidget:
    """One widget of the native project tree (page, graph, axis, xy, ...).

    Mirrors the subset of the veusz.embed widget API the AutoPlot programs
    use - ``Add``, ``Remove``, ``WalkWidgets``, attribute access to children
    and settings - so the same page-building code runs against a NativeDocument
    (written with vszh5_writer, no Veusz needed) or a live Veusz document.
    """
    def __init__(self, document, kind, name, parent=None):
        """Create a widget of ``kind`` named ``name`` under ``parent``.

        ``settings`` holds the explicit setting values (path -> value) in the order
        they were set; ``children`` the child widgets in drawing order.
        """
        self.document = document
        self.kind = kind
        self.name = name
        self.parent = parent
        self.children = []
        self.settings = OrderedDict()
        self.auto_named = False
        self.time_source = None        # 0.5.0: (x, y) datasets before the time layout
        self.time_layout_key = None    # 0.5.0: TimeLayout applied to this axis       # 0.3.0: named from its data by finalize()
        self.local_time = None        # 0.3.0: label prefix of a local date-time axis

    @property
    def typename(self):
        """Widget type, as veusz.embed names it."""
        return self.kind

    @property
    def childnames(self):
        """Names of the child widgets, as veusz.embed names them."""
        return [c.name for c in self.children]

    @property
    def path(self):
        """Absolute widget path, e.g. ``/page_1/graph1/x``; ``/`` for the root.
        """
        if self.parent is None:
            return "/"
        return self.parent.path.rstrip("/") + "/" + self.name

    def __getattr__(self, name):
        """Child widget called ``name`` if there is one, else a setting proxy.

        As in veusz.embed, ``page.graph1`` returns the child widget and
        ``graph1.Border.hide`` the setting proxy.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        for child in self.children:
            if child.name == name:
                return child
        return _Setting(self, name)

    def Add(self, kind, name=None, autoadd=True, **settings):
        """Add a child widget of ``kind`` and return it.

        ``name`` defaults to the kind; a clash gets a numeric suffix (graph,
        graph_1, graph_2, ...).  A widget added without a name is marked
        ``auto_named`` so finalize() may rename it after its data.  Keyword
        arguments become settings.  With ``autoadd`` a new graph gets its x
        and y axes, as Veusz does; pass ``autoadd=False`` to build the axes
        yourself.  Unknown kinds and invalid names raise ValueError.
        """
        if kind not in WIDGET_KINDS:
            raise ValueError("Unsupported native widget: %s" % kind)
        existing = {c.name for c in self.children}
        stem = name or kind
        candidate = stem
        index = 1
        while candidate in existing:
            candidate = "%s_%d" % (stem, index)
            index += 1
        if not candidate or "/" in candidate or candidate in (".", ".."):
            raise ValueError("Invalid widget name: %r" % candidate)
        widget = NativeWidget(self.document, kind, candidate, self)
        widget.auto_named = name is None          # renamed from its data by finalize()
        for key, value in settings.items():
            widget.settings[key] = _plain(value)
        self.children.append(widget)
        if kind == "graph" and autoadd:
            widget.Add("axis", name="x", direction="horizontal")
            widget.Add("axis", name="y", direction="vertical")
        return widget

    def Remove(self, name=None):
        """Remove this widget (``name`` None) or its child called ``name``.

        The document root cannot be removed; an unknown child name raises
        KeyError.
        """
        if name is None:
            if self.parent is None:
                raise ValueError("Cannot remove document root")
            self.parent.children.remove(self)
            return
        for child in self.children:
            if child.name == name:
                self.children.remove(child)
                return
        raise KeyError(name)

    def WalkWidgets(self):
        """Yield this widget and every descendant, depth first, in drawing order.
        """
        yield self
        for child in self.children:
            yield from child.WalkWidgets()

    def _write(self, doc):
        """Emit this widget's settings and children into a vszh5_writer.VszDoc."""
        if self.kind == "graph" and "bottomMargin" not in self.settings:
            if any(child.kind in ("axis", "axis-broken") and
                   child.settings.get("mode") == "datetime"
                   for child in self.children):
                doc.set("bottomMargin", "3cm")
        if self.kind in ("axis", "axis-broken") and self.settings.get("mode") == "datetime":
            if "TickLabels/size" not in self.settings:
                doc.set("TickLabels/size", "8pt")
            if "Label/size" not in self.settings:          # 0.5.0: long ISO labels
                doc.set("Label/size", "10pt")
        for key, value in self.settings.items():
            # AutoPlot captions are plain text, not Veusz math/markup:
            # vz_text is Veusz's own latexEscape.  Dataset names and dataset
            # contents are never rewritten.
            if key in ("label", "key") and isinstance(value, str):
                value = vz_text(value)
            doc.set(key, value)
        for child in self.children:
            doc.add(child.kind, child.name)
            doc.to(child.name)
            child._write(doc)
            doc.up()


def mark_local_time_axis(axis, label="Local time"):
    """
    Make ``axis`` (a NativeWidget axis or axis-broken) a local date-time axis
    (0.3.0): its tick format and label are set from the local datasets drawn
    on it by NativeDocument.finalize() (ISO 8601, 24 h, with the offset).
    """
    axis.local_time = label
    axis.settings["mode"] = "datetime"
    axis.settings["autoRange"] = axis.settings.get("autoRange", "exact")
    return axis


# %% Time axes: local ISO 8601, line breaks and axis breaks (0.5.0)
class TimeLayout(object):
    """
    Gap rule of the time axes (0.5.0), see vszh5_writer TIME GAPS.

    Small gap - the trace line breaks, the axis stays continuous:
        dt > k * median(dt) of that trace, or dt > manual_s when manual_s > 0.
    Large gap - the x axis itself breaks (Veusz axis-broken), where no trace
    on the axis has data for
        dt > n * (largest small-gap threshold of the traces on that axis),
        or dt > axis_manual_s when axis_manual_s > 0;
    n = 0 with axis_manual_s = 0: no axis breaks. A large-gap threshold is
    never smaller than the small one, so an axis break is always also a
    line break.
    """

    def __init__(self, k=vszh5_writer.GAP_K, n=vszh5_writer.GAP_AXIS_N, manual_s=0.0,
                 axis_manual_s=0.0):
        """Store the gap settings (negative values are clamped to 0).

        k: small-gap factor - a trace's line breaks where its step dt exceeds
        k x median(dt) of that trace.
        n: large-gap factor - the x axis breaks where no trace has data for
        dt > n x the largest small-gap threshold; 0 disables axis breaks.
        manual_s / axis_manual_s: fixed small / large gap in seconds; > 0
        overrides k / n.
        """
        self.k = float(k)
        self.n = max(0.0, float(n))
        self.manual_s = max(0.0, float(manual_s or 0.0))
        self.axis_manual_s = max(0.0, float(axis_manual_s or 0.0))

    def key(self):
        """Tuple of the settings; a change re-lays out the time axes on save.
        """
        return (self.k, self.n, self.manual_s, self.axis_manual_s)

    def axis_threshold(self, line_threshold):
        """Large-gap threshold for an axis whose largest small-gap threshold
        is ``line_threshold`` (seconds); 0.0 = no axis breaks."""
        if self.axis_manual_s > 0:
            return max(self.axis_manual_s, float(line_threshold))
        if self.n > 0 and line_threshold > 0:
            return max(self.n, 1.0) * float(line_threshold)
        return 0.0

    def describe(self):
        """One-line human description of the gap rule, written into graph notes.
        """
        line = ("dt > %s" % _seconds_text(self.manual_s) if self.manual_s > 0
                else "dt > %g x median(dt)" % self.k)
        if self.axis_manual_s > 0:
            axis = "x-axis break (large gap) where dt > %s" % _seconds_text(self.axis_manual_s)
        elif self.n > 0:
            axis = "x-axis break (large gap) where dt > %g x the small-gap threshold" % self.n
        else:
            axis = "no x-axis breaks"
        return "Gaps: line break (small gap) where %s; %s." % (line, axis)


def _seconds_text(seconds):
    """A threshold for people: '45 s', '12 min', '3.5 h', '2.1 d'."""
    seconds = float(seconds)
    for size, unit in ((86400.0, "d"), (3600.0, "h"), (60.0, "min")):
        if seconds >= size:
            return "%g %s" % (round(seconds / size, 3), unit)
    return "%g s" % round(seconds, 3)


#: MJD of the Veusz date origin 2009-01-01T00:00:00 UTC.
MJD_VEUSZ_EPOCH = 54832.0
#: Plausible MJD range of a time column (1941 .. 2132); values outside are
#: not taken for MJD (and MJD 0 padding rows are ignored).
MJD_RANGE = (30000.0, 100000.0)
_TIME_NAME_HINT = re.compile(r"(^|[^A-Za-z])(D?MJD|JD|TIME|DATE)([^A-Za-z]|$)|__x__", re.I)
_SECONDS_NAMES = frozenset(("TIME", "TIMESTAMP", "TIMESTAMP_SEC", "SECONDS", "SEC",
                            "UT_SECONDS", "UTSECONDS", "ELAPSED", "ELAPSED_S", "T_SEC"))


def _time_kind(doc, name, axis):
    """'date' (Veusz date dataset, UTC), 'mjd', 'seconds' or None (not time)."""
    entry = doc.datasets.get(name) if isinstance(name, str) else None
    if entry is None:
        return None
    kind, parts = entry
    if kind == "date":
        return "date"
    if kind != "1d":
        return None
    values = np.asarray(parts["data"], dtype=float)
    finite = values[np.isfinite(values) & (values > 0)]     # MJD 0 = padding row
    label = str(axis.settings.get("label", "") or "")
    hinted = bool(_TIME_NAME_HINT.search(name) or _TIME_NAME_HINT.search(label))
    if (hinted and finite.size and np.all(finite >= MJD_RANGE[0])
            and np.all(finite <= MJD_RANGE[1])):
        return "mjd"
    if name.split("__")[-1].upper() in _SECONDS_NAMES or label.strip().upper() in _SECONDS_NAMES:
        return "seconds"
    return None


def _plot_name(doc, wanted, owner):
    """A dataset name for a plot copy: ``wanted``, or wanted_2, ... when
    another source already uses that name (owner identifies the source)."""
    used = doc._plot_copies
    name, number = wanted, 2
    while name in used and used[name] != owner or (name in doc.datasets and name not in used):
        name = "%s_%d" % (wanted, number)
        number += 1
    used[name] = owner
    return name


def _layout_time_axis(doc, graph, axis):
    """
    Lay out one x axis whose traces are against time (0.5.0).

    MJD or Veusz-date x data are stored once as a UTC date dataset
    ("<x>_utc" for MJD) and drawn from a local copy "<utc>_local" (this
    computer's clock, DST-aware, ISO 8601 ticks with the UTC offset). A
    trace whose step exceeds the line-break threshold gets NaN rows there
    (plot copies "<utc>_local" and "<y>_gaps"; the stored data stay
    unchanged). The axis becomes axis-broken only for large gaps (n > 0).
    """
    layout = getattr(doc, "time_layout", None) or TimeLayout()
    if not hasattr(doc, "_plot_copies"):
        doc._plot_copies = {}
    xys = [c for c in graph.WalkWidgets()
           if c.kind == "xy" and c.settings.get("xAxis", "x") == axis.name]
    if not xys:
        return
    if (axis.time_layout_key == layout.key()
            and all(xy.time_source is not None for xy in xys)):
        return
    reverse_local = dict((v, k) for k, v in doc.local_of.items())
    plans = []
    for xy in xys:
        x_name, y_name = xy.time_source or (xy.settings.get("xData"), xy.settings.get("yData"))
        x_name = reverse_local.get(x_name, x_name)
        kind = _time_kind(doc, x_name, axis)
        if kind is None or not isinstance(y_name, str) or y_name not in doc.datasets:
            plans.append(None)
            continue
        plans.append((xy, x_name, y_name, kind))
    timed = [p for p in plans if p is not None]
    if not timed:
        return
    kinds = set(p[3] for p in timed)
    dated = kinds <= set(("date", "mjd"))
    local_names, union, thresholds = [], [], []
    for xy, x_name, y_name, kind in timed:
        values = np.asarray(doc.datasets[x_name][1]["data"], dtype=float)
        if kind == "mjd":
            # Veusz date seconds = (MJD - 54832) * 86400 (54832 = 2009-01-01).
            # The MJD dataset stays the stored value; the plot copy below is
            # the Veusz date-time dataset (a tool stores a UTC date dataset
            # of its raw time column itself, once per file - not per trace).
            utc = (values - MJD_VEUSZ_EPOCH) * 86400.0
            utc[(values < MJD_RANGE[0]) | (values > MJD_RANGE[1])] = np.nan
            utc_name = x_name
        elif kind == "date":
            utc, utc_name = values, x_name
        else:
            utc, utc_name = values, x_name
        seconds = utc                                  # gap rule in seconds
        threshold = vszh5_writer.gap_threshold(seconds, layout.k, layout.manual_s)
        cut = vszh5_writer.gap_positions(seconds, threshold)
        thresholds.append(threshold)
        if dated:
            local = vszh5_writer.local_time_from_veusz(utc)
            local_name = utc_name + "_local"
            doc.SetDataDateTime(local_name, vszh5_writer.insert_gaps(local.veusz_seconds, cut))
            doc.local_axes[local_name] = local
            doc.local_of[utc_name] = local_name
            plot_x = local_name
            union.append(local.veusz_seconds)
            if local_name not in local_names:
                local_names.append(local_name)
        else:
            plot_x = x_name
            if cut.size:
                plot_x = _plot_name(doc, x_name + "_gaps", x_name)
                doc.SetData(plot_x, vszh5_writer.insert_gaps(values, cut))
            union.append(values)
        plot_y = y_name
        if cut.size:
            y_values = doc.datasets[y_name][1]["data"]
            if np.asarray(y_values).shape == values.shape:
                plot_y = _plot_name(doc, y_name + "_gaps", (x_name, y_name))
                doc.SetData(plot_y, vszh5_writer.insert_gaps(y_values, cut))
            else:
                # y does not line up with x: draw it unbroken against a
                # local copy without gap rows (never misalign the points).
                if dated:
                    plot_x = _plot_name(doc, utc_name + "_local_nogaps", utc_name)
                    doc.SetDataDateTime(plot_x, local.veusz_seconds)
                    doc.local_axes[plot_x] = local
                else:
                    plot_x = x_name
        xy.time_source = (x_name, y_name)
        xy.settings["xData"] = plot_x
        xy.settings["yData"] = plot_y
    # Large gaps break the x axis: only where no trace has data for longer
    # than the large-gap threshold (n x the largest small-gap threshold, or
    # the fixed large gap).
    pairs = []
    axis_threshold = layout.axis_threshold(max(thresholds) if thresholds else 0.0)
    if axis_threshold > 0:
        pairs = vszh5_writer.axis_break_pairs(np.concatenate(union), axis_threshold)
    if pairs:
        axis.kind = "axis-broken"
        axis.settings["breakPoints"] = [float(v) for pair in pairs for v in pair]
    else:
        axis.kind = "axis"
        axis.settings.pop("breakPoints", None)
    if dated:
        fmt, text = doc.local_axis_text(local_names, "Local time")
        axis.settings["mode"] = "datetime"
        axis.settings["TickLabels/format"] = fmt
        axis.settings.setdefault("TickLabels/rotate", str(vszh5_writer.LOCAL_DATETIME_ROTATE))
        axis.settings["label"] = text
        axis.settings.setdefault("autoRange", "exact")
    axis.time_layout_key = layout.key()
    note = layout.describe()
    if note not in str(graph.settings.get("notes", "")):
        graph.settings["notes"] = (str(graph.settings.get("notes", "")) + "\n" + note).strip()


def _size_dated_page(page):
    """
    Give a page with date-time axes room for its ISO 8601 tick labels (0.5.0).

    Only pages without an explicit size: Veusz's default 15 x 15 cm page
    leaves a grid of graphs no room for 25-character rotated tick labels.
    A grid gets 13 cm per column and 8 cm per row; a single graph 25 x 16 cm.
    """
    if "width" in page.settings or "height" in page.settings:
        return
    if not any(w.kind in ("axis", "axis-broken") and w.settings.get("mode") == "datetime"
               for w in page.WalkWidgets()):
        return
    graphs = [w for w in page.WalkWidgets() if w.kind == "graph"]
    grids = [w for w in page.children if w.kind == "grid"]
    if grids and graphs:
        try:
            columns = max(1, int(grids[0].settings.get("columns", 2) or 2))
        except (TypeError, ValueError):
            columns = 2
        rows = int(math.ceil(len(graphs) / float(columns)))
        page.settings["width"] = "%dcm" % (13 * columns)
        page.settings["height"] = "%dcm" % max(16, 8 * rows)
    else:
        page.settings["width"] = "25cm"
        page.settings["height"] = "16cm"


# %% Smith charts (0.4.0)
def apply_settings(widget, settings):
    """
    Set ``{setting path: value}`` on a widget (0.4.0).

    A NativeWidget stores the values directly; any other widget (a
    veusz.embed node) is reached by walking the path as attributes
    (``"PlotLine/color"`` -> ``widget.PlotLine.color.val``), so the same
    specification builds a Veusz-embedded document too.
    """
    if isinstance(widget, NativeWidget):
        for path, value in settings.items():
            widget.settings[path] = _plain(value)
        return widget
    for path, value in settings.items():
        node = widget
        for part in path.split("/"):
            node = getattr(node, part)
        node.val = value
    return widget


def add_smith_graph(parent, grid, traces, name=None, notes="", key="auto",
                    key_position=("outside", "top"), key_width="6.5cm",
                    limit=vszh5_writer.SMITH_LIMIT):
    """
    Add a Smith chart graph (vszh5_writer.smith_graph_spec) under ``parent``
    (a page or grid widget); returns the graph widget (0.4.0).

    ``grid`` is a vszh5_writer.SmithGrid whose datasets are written to the
    same document (``grid.write(doc)``); ``traces`` are
    vszh5_writer.smith_trace() traces.  Captions are passed unescaped:
    NativeWidget escapes "label" and "key" text when it writes the file.
    """
    graph_settings, children = vszh5_writer.smith_graph_spec(
        grid, traces, key=key, key_position=key_position, limit=limit, escape=False,
        notes=notes, key_width=key_width)
    graph = parent.Add("graph", name=name or "smith", autoadd=False)
    apply_settings(graph, graph_settings)
    for kind, child_name, settings in children:
        apply_settings(graph.Add(kind, name=child_name), settings)
    return graph


def add_smith_page(root, page_name, grid, traces, title="", notes="", graph_name="",
                   graph_notes="", size="18cm", key="auto", key_position=("outside", "top"),
                   key_width="6.5cm"):
    """
    A page with one Smith chart, as vszh5_writer.smith_page() builds it, on a
    NativeDocument (or veusz.embed) widget tree; returns the graph (0.4.0).

    ``notes`` are the page notes (source header); ``graph_notes`` are added
    below the chart type in the graph notes (e.g. "Z0 = 50 ohm").
    """
    width, height, top_margin = vszh5_writer.smith_page_layout(
        traces, title, size=size, key=key, key_position=key_position, key_width=key_width)
    page = root.Add("page", name=page_name)
    apply_settings(page, {"width": width, "height": height})
    if notes:
        apply_settings(page, {"notes": notes})
    if title:
        apply_settings(page.Add("label", name="title"),
                       vszh5_writer.title_settings(title, escape=False))
    graph = add_smith_graph(page, grid, traces,
                            name=graph_name or vszh5_writer.widget_name(page_name, prefix="g_"),
                            notes=graph_notes, key=key, key_position=key_position,
                            key_width=key_width)
    apply_settings(graph, {"topMargin": top_margin})
    return graph


_DATA_SETTING_OF = {"xy": "yData", "nonorthpoint": "data1", "image": "data",
                    "bar": "lengths", "function": "function"}


def _data_name(widget):
    """The dataset a plotting widget draws (y for xy), or ''."""
    value = widget.settings.get(_DATA_SETTING_OF.get(widget.kind, ""), "")
    if isinstance(value, (tuple, list)):
        value = value[0] if value else ""
    return value if isinstance(value, str) else ""


def _rename_from_data(widget):
    """Name an auto-named widget after its data (unique among siblings)."""
    parent = widget.parent
    if parent is None:
        return
    if widget.kind in _DATA_SETTING_OF:
        data = _data_name(widget)
        base = vszh5_writer.widget_name(data) if data else ""
    elif widget.kind in ("graph", "polar"):
        first = next((_data_name(c) for c in widget.WalkWidgets()
                      if c is not widget and _data_name(c)), "")
        base = vszh5_writer.widget_name(first, prefix="g_") if first else ""
    else:
        base = ""
    widget.auto_named = False
    if not base or base == widget.name:
        return
    used = {c.name for c in parent.children if c is not widget}
    widget.name = vszh5_writer.unique_widget_name(base, used)


# %% Dataset model and file output
class NativeDocument:
    """Build and atomically save a self-contained Veusz format-1 document."""

    def __init__(self, name="AutoPlot"):
        """Empty project: a document root, no datasets, default TimeLayout.

        ``local_axes`` maps each local date-time display dataset to its
        vszh5_writer.LocalTimeAxis (tick format and label); ``local_of`` maps a
        UTC date dataset to its local copy.
        """
        self.name = name
        self.Root = NativeWidget(self, "document", "")
        self.datasets = OrderedDict()
        self.tags = {}
        self.local_axes = {}          # local dataset name -> vszh5_writer.LocalTimeAxis
        self.local_of = {}            # UTC date dataset name -> its local copy
        # 0.5.0: gap rule of the time axes resolved by finalize():
        # line break when dt > k * median(dt) (or > manual_s seconds),
        # axis break when dt > n * k * median(dt); n = 0: one axis.
        self.time_layout = TimeLayout()

    def _put(self, name, kind, parts):
        """Store dataset ``name`` of ``kind`` ('1d', 'date', 'text', '2d') with its
        arrays ``parts``; rejects empty names, NUL characters, '.' and '..'.
        """
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ValueError("Dataset name must be nonempty text without NUL")
        if name in (".", ".."):
            raise ValueError("Dataset name cannot be '.' or '..'")
        self.datasets[name] = (kind, parts)

    def SetData(self, name, data=None, symerr=None, negerr=None, poserr=None, val=None):
        """1-D numeric dataset (``val=`` accepted as veusz.embed spells it)."""
        if data is None:
            data = val
        parts = {}
        for key, values in (("data", data), ("serr", symerr),
                            ("nerr", negerr), ("perr", poserr)):
            if values is not None:
                array = np.array(values, dtype=np.float64, copy=True)
                if array.ndim == 0:
                    array = array.reshape(1)
                if array.ndim != 1:
                    raise ValueError("%s/%s requires a 1-D array" % (name, key))
                if key != "data" and array.shape != parts["data"].shape:
                    raise ValueError("Error array length mismatch: %s" % name)
                parts[key] = array
        self._put(name, "1d", parts)

    def SetDataText(self, name, values):
        """Store a text dataset; bytes are decoded as UTF-8 (bad bytes replaced).

        NUL characters are rejected because HDF5 strings cannot hold them.
        """
        data = [v.decode("utf-8", "replace") if isinstance(v, bytes)
                else str(v) for v in values]
        if any("\x00" in v for v in data):
            raise ValueError("Text datasets cannot contain NUL")
        self._put(name, "text", {"data": data})

    def SetData2D(self, name, data, xrange=None, yrange=None,
                  xedge=None, yedge=None, xcent=None, ycent=None):
        """Store a 2-D (image) dataset with optional ranges, edges or centres.

        ``data`` is copied as float64 and must be 2-D; ``xrange``/``yrange`` are
        (min, max) pairs, ``xedge``/``yedge`` bin edges and ``xcent``/``ycent``
        bin centres, exactly as Veusz's SetData2D takes them.
        """
        array = np.array(data, dtype=np.float64, copy=True)
        if array.ndim != 2:
            raise ValueError("%s requires a 2-D array" % name)
        parts = {"data": array}
        for key, value in (("xrange", xrange), ("yrange", yrange),
                           ("xedge", xedge), ("yedge", yedge),
                           ("xcent", xcent), ("ycent", ycent)):
            if value is not None:
                parts[key] = np.array(value, dtype=np.float64, copy=True)
        self._put(name, "2d", parts)

    def SetDataDateTime(self, name, seconds):
        """Date dataset: seconds since 2009-01-01 00:00:00 UTC (Veusz date-time)."""
        self.SetData(name, seconds)
        self.datasets[name] = ("date", self.datasets[name][1])

    def SetDataLocalTime(self, utc_name, local_name=None, zone=None, label="Local time"):
        """
        Local display copy of the UTC date dataset ``utc_name`` (0.3.0).

        Writes ``local_name`` (default "<utc_name>_local") as a date dataset
        on the processing computer's clock (or ``zone``; VSZH5_DISPLAY_TZ
        overrides), DST-aware, and remembers its offsets so a marked axis
        gets an ISO-8601 tick format with the offset.  Returns the name.
        """
        kind, parts = self.datasets[utc_name]
        if kind not in ("date", "1d"):
            raise ValueError("%s is not a date dataset" % utc_name)
        axis = vszh5_writer.local_time_from_veusz(parts["data"], zone=zone, label=label)
        local_name = local_name or "%s_local" % utc_name
        self.SetDataDateTime(local_name, axis.veusz_seconds)
        self.local_axes[local_name] = axis
        self.local_of[utc_name] = local_name
        return local_name

    def local_axis_text(self, dataset_names, label="Local time"):
        """(tick format, axis label) of an axis showing these local datasets."""
        axes = [self.local_axes[n] for n in dataset_names if n in self.local_axes]
        if not axes:
            return vszh5_writer.ISO_DATETIME_FORMAT, label
        offsets = []
        for axis in axes:
            # 0.5.0: only instants that exist (a NaN time has offset 0).
            finite = axis.offsets_s[np.isfinite(axis.veusz_seconds) & np.isfinite(axis.offsets_s)]
            for value in finite.tolist():
                if value not in offsets:
                    offsets.append(value)
        offsets = offsets or [0.0]
        if len(offsets) == 1:
            text = vszh5_writer.iso_offset_text(offsets[0])
            return (vszh5_writer.ISO_DATETIME_FORMAT + text,
                    "%s (ISO 8601, UTC%s)" % (label, text))
        changes = []
        for axis in axes:
            for instant, offset in axis.changes:
                if (instant, offset) not in changes:
                    changes.append((instant, offset))
        changes.sort()
        # 0.5.0: compact for grid panels - the first offset, then each real
        # change of offset (every trace starts a "change" of its own, so
        # equal neighbours are merged) with the local minute it starts.
        parts, current = [], None
        for instant, offset in changes:
            if offset == current:
                continue
            parts.append("UTC%s" % offset if current is None
                         else "UTC%s from %s" % (offset, instant[:16]))
            current = offset
        return (vszh5_writer.ISO_DATETIME_FORMAT,
                "%s (ISO 8601; %s)" % (label, ", ".join(parts)))

    def finalize(self):
        """Name unnamed widgets, then lay out every time axis: local ISO 8601
        date-times, line breaks at gaps and (optionally) axis breaks (0.5.0;
        0.3.0 resolved marked local-time axes only)."""
        for widget in list(self.Root.WalkWidgets()):
            if widget.auto_named:
                _rename_from_data(widget)
        for graph in [w for w in self.Root.WalkWidgets() if w.kind == "graph"]:
            for axis in [c for c in graph.children if c.kind in ("axis", "axis-broken")]:
                if axis.settings.get("direction", "horizontal") != "vertical":
                    _layout_time_axis(self, graph, axis)
        for page in [w for w in self.Root.children if w.kind == "page"]:
            _size_dated_page(page)
        for widget in self.Root.WalkWidgets():
            label = widget.local_time
            if label is None or widget.kind not in ("axis", "axis-broken"):
                continue
            if widget.time_layout_key is not None:      # done by _layout_time_axis
                continue
            graph = widget.parent
            names = []
            for child in graph.WalkWidgets() if graph is not None else ():
                if child.kind == "xy":
                    x_name = child.settings.get("xData")
                    if x_name in self.local_of:          # UTC date -> local copy
                        x_name = self.local_of[x_name]
                        child.settings["xData"] = x_name
                    if isinstance(x_name, str) and x_name not in names:
                        names.append(x_name)
            fmt, text = self.local_axis_text(names, label)
            widget.settings["mode"] = "datetime"
            widget.settings["TickLabels/format"] = fmt
            widget.settings.setdefault("TickLabels/rotate", str(vszh5_writer.LOCAL_DATETIME_ROTATE))
            widget.settings["label"] = text

    def TagDatasets(self, tag, names):
        """Attach ``tag`` to the existing datasets ``names`` (Veusz dataset tags).

        The tag must be a valid HDF5 name (no '/', NUL, '.' or '..'); unknown
        dataset names raise KeyError before anything is changed.
        """
        if not tag or "/" in tag or "\x00" in tag or tag in (".", ".."):
            raise ValueError("Unsupported HDF5 tag name: %r" % tag)
        names = set(names)
        missing = names - set(self.datasets)
        if missing:
            raise KeyError("Unknown tagged datasets: %s" % sorted(missing))
        self.tags.setdefault(tag, set()).update(names)

    def GetDatasets(self):
        """Names of all datasets, in the order they were created."""
        return list(self.datasets)

    def GetData(self, name):
        """Arrays of dataset ``name``: (data, serr, nerr, perr) for numeric and
        date datasets, (data,) for text and 2-D datasets.
        """
        kind, parts = self.datasets[name]
        if kind in ("1d", "date"):
            return tuple(parts.get(k) for k in ("data", "serr", "nerr", "perr"))
        return (parts["data"],)

    def Get(self, path):
        """Value of the setting at ``path`` (``/page/graph/x/label``), or None.

        Walks the widget tree by name; the first path part that is not a child
        widget starts the setting path.
        """
        nodes = path.strip("/").split("/")
        widget = self.Root
        while nodes:
            match = next((c for c in widget.children if c.name == nodes[0]), None)
            if match is None:
                return widget.settings.get("/".join(nodes))
            widget = match
            nodes.pop(0)
        raise KeyError(path)

    def referenced_datasets(self):
        """Exact references from the model; no RPC, regex or destructive pruning."""
        used = set()
        for widget in self.Root.WalkWidgets():
            for key, value in widget.settings.items():
                if key.split("/")[-1] not in DATASET_SETTINGS:
                    continue
                for item in value if isinstance(value, (tuple, list)) else [value]:
                    if isinstance(item, str) and item:
                        if item not in self.datasets:
                            raise ValueError("Missing dataset %r at %s/%s" %
                                             (item, widget.path, key))
                        used.add(item)
        return used

    def SaveToString(self):
        """The Veusz command script, built with vszh5_writer.VszDoc."""
        doc = VszDoc(generator="AutoPlot _autoplot_native %s, vszh5_writer %s"
                     % (VERSION, WRITER_VERSION))
        # Compatibility level of the settings baseline (Veusz 3.4 / 4.1).
        doc.lines.append("SetCompatLevel(0)")
        self.finalize()
        self.Root._write(doc)
        return doc.text()

    GetSaveString = SaveToString

    def save(self, filename, minimized=False):
        """Write the document with vszh5_writer.VeuszWriter (atomic).

        The file is written to a private same-directory ".part" file and
        moved onto ``filename`` only when everything succeeded; any error
        deletes the partial file and leaves an existing ``filename`` as it
        was.  ``minimized`` keeps only the datasets the widgets use.
        """
        filename = os.fspath(filename)
        if not filename.lower().endswith(".vszh5"):
            filename = os.path.splitext(filename)[0] + ".vszh5"
        self.finalize()
        used = self.referenced_datasets()  # validate even a full save
        names = used if minimized else set(self.datasets)
        script = self.SaveToString()
        tags_of = {}
        for tag, members in self.tags.items():
            for member in members:
                tags_of.setdefault(member, []).append(tag)
        with VeuszWriter(filename, compress=True,
                         writer_tag="AutoPlot _autoplot_native %s" % VERSION) as writer:
            for name in sorted(names):
                kind, parts = self.datasets[name]
                tags = sorted(tags_of.get(name, ()))
                if kind == "1d":
                    writer.dataset_1d(name, parts["data"], serr=parts.get("serr"),
                                      perr=parts.get("perr"), nerr=parts.get("nerr"),
                                      tags=tags)
                elif kind == "date":
                    writer.dataset_date(name, parts["data"], tags=tags)
                elif kind == "text":
                    writer.dataset_text(name, parts["data"], tags=tags)
                elif kind == "2d":
                    axes = {k: v for k, v in parts.items() if k != "data"}
                    writer.dataset_2d(name, parts["data"], tags=tags, **axes)
                else:  # pragma: no cover - _put only stores the four kinds
                    raise ValueError("Unknown dataset kind %r" % kind)
            writer.finish(script)
        return filename

    def Save(self, filename, mode="hdf5"):
        """veusz.embed-compatible alias of ``save``; only mode 'hdf5' is supported.
        """
        if mode != "hdf5":
            raise ValueError("Only native HDF5 output is supported")
        return self.save(filename)

    def Close(self):
        """No external process to close; keep data alive until object disposal."""
