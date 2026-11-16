# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
docs/writer_0.0.30/make_0030.py

Turn vszh5_writer master 0.0.29 into 0.0.30 (gap options for every plot):
the version lines, the status / lineage / differences text, and the 0.0.30
additions (GapRule, rule_gap_positions, rule_axis_break_pairs, break_points,
gap_plan, local_veusz_exact, axis_break_pairs(log_threshold=), x_breaks of
_add_axes / xy_page / Panel / time_pair_pages).  Additions only: every
0.0.29 call writes the same file.

Usage:  python make_0030.py vszh5_writer_0.0.29.py OUT.py
        python add_entry_0030.py OUT.py
The input must be the registry master 0.0.29 (SHA-256 checked); every
replacement must match exactly once.  add_entry_0030.py then inserts the
revision entry and checks that OUT.py is the 0.0.30 master byte for byte.

Created on 2026-11-16

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: First version (FitsAutoPlot suite 0.4.3).
Date: 2026-11-16
# %%%%% Function Descriptions
        once: Replace ``old`` by ``new``; ``old`` must occur exactly once.
# %%%%% Variable Descriptions
    Define all utilized variables
        MASTER_0029: SHA-256 of the registry master 0.0.29
        src_path, out_path: input (0.0.29) and output (0.0.30) files
        raw, src: input bytes and text being edited
# %%%%% More Info
    The two scripts and vszh5_writer_0.0.29.py rebuild vszh5_writer.py
    0.0.30 exactly (SHA-256
    26194c8e96572fce2106bb5588d33d7c3f024fbd5169ff626cab2e40aa01f604).
    The difference is vszh5_writer_0.0.29_to_0.0.30.diff.  Registry: RF
    project claude/vszh5-writer-registry.md.
=============================================================================
"""
import hashlib
import sys

MASTER_0029 = "50631331470ebd50162030f971529044076a6c813a264393067522a504f9d701"
src_path, out_path = sys.argv[1], sys.argv[2]
raw = open(src_path, "rb").read()
assert hashlib.sha256(raw).hexdigest() == MASTER_0029, "input is not master 0.0.29"
src = raw.decode("utf-8")


def once(old, new):
    """Replace ``old`` by ``new``; ``old`` must occur exactly once."""
    global src
    found = src.count(old)
    assert found == 1, (found, old[:90])
    src = src.replace(old, new)


# ---- version ------------------------------------------------------------
once('__version__ = "0.0.29"                 # Semantic Versioning 2.0.0',
     '__version__ = "0.0.30"                 # Semantic Versioning 2.0.0')
once(" Version        : 0.0.29 (Semantic Versioning 2.0.0, MAJOR.MINOR.PATCH)",
     " Version        : 0.0.30 (Semantic Versioning 2.0.0, MAJOR.MINOR.PATCH)")
once(" Modified       : 2026-11-10", " Modified       : 2026-11-16")
once("""                  save: files identical to 0.0.27; 0.0.29 exact ISO text:
                  3.6.2, every other output identical to 0.0.28)""",
     """                  save: files identical to 0.0.27; 0.0.29 exact ISO text:
                  3.6.2, every other output identical to 0.0.28; 0.0.30
                  gap options: 4.2.1, every 0.0.29 call identical)""")
once("""                  ISO-8601 text without dropping digits on request
                  (iso_local_strings timespec="exact"). One master copy: the""",
     """                  ISO-8601 text without dropping digits on request
                  (iso_local_strings timespec="exact"); 0.0.30 (FitsAutoPlot
                  suite 0.4.3, for every plot generator and the MHM2010 /
                  T4 workbench) adds the gap options: GapRule (line breaks
                  and / or x-axis breaks, either one off, a log-scale check
                  for frequency sweeps), gap_plan, x_breaks= on the page
                  helpers, local_veusz_exact. One master copy: the""")
once("""  arithmetic (MJD 61234.10355 -> 22:29:06.72, not 06.7199999...).

# %%% References""",
     """  arithmetic (MJD 61234.10355 -> 22:29:06.72, not 06.7199999...).
* (0.0.30) Gap options for any plotted series: GapRule (k, n, fixed small
  and large gaps, break_lines and break_axis - either can be off - and a
  geometric check so a log-spaced frequency sweep is not called gapped),
  rule_gap_positions, rule_axis_break_pairs, gap_plan (gap rows for each
  trace and the break points of the shared x axis in one call),
  break_points; axis_break_pairs(log_threshold=); xy_page(x_breaks=),
  Panel.x_breaks and time_pair_pages(mjd_breaks=, local_breaks=) draw a
  Veusz axis-broken x axis; local_veusz_exact (UTC + offset in one step,
  no round trip through Unix seconds).

# %%% References""")

# ---- axis_break_pairs: optional log-scale condition ----------------------
once('''def axis_break_pairs(t: Any, threshold: float,
                     margin: float = GAP_AXIS_MARGIN) -> list[tuple[float, float]]:''',
     '''def axis_break_pairs(t: Any, threshold: float,
                     margin: float = GAP_AXIS_MARGIN,
                     log_threshold: float = 0.0) -> list[tuple[float, float]]:''')
once('''    0.0.27: the shrink is relative to the gap; an absolute 1e-9 x |t| (the
    FitsAutoPlot rule before) is about 5 s at MJD 61000, which swallowed
    short gaps on MJD axes.
    """
    if not threshold or threshold <= 0.0:''',
     '''    0.0.27: the shrink is relative to the gap; an absolute 1e-9 x |t| (the
    FitsAutoPlot rule before) is about 5 s at MJD 61000, which swallowed
    short gaps on MJD axes.
    0.0.30: ``log_threshold`` > 0 (natural-log units) also requires
    log(stop / start) > log_threshold, so the long steps at the top of a
    log-spaced frequency sweep are not cut out (only when every value is
    positive; the default 0 keeps the 0.0.27 result).
    """
    if not threshold or threshold <= 0.0:''')
once('''    steps = np.diff(values)
    pairs = []
    for index in np.flatnonzero(steps > threshold):
        start, stop = float(values[index]), float(values[index + 1])''',
     '''    steps = np.diff(values)
    large = steps > threshold
    if log_threshold and log_threshold > 0.0 and values[0] > 0.0:
        large &= np.diff(np.log(values)) > float(log_threshold)
    pairs = []
    for index in np.flatnonzero(large):
        start, stop = float(values[index]), float(values[index + 1])''')

# ---- new section after axis_break_pairs --------------------------------
NEW_SECTION = r'''

# %% ===========================================================================
#    GAP OPTIONS FOR ANY PLOTTED SERIES (0.0.30)
# =============================================================================
# The 0.0.27 rule, with the two consequences of a gap made optional and the
# rule usable for any sorted x - time, frequency, angle, index:
#
#   line break (small gap):  dx_i > tau = K * median(dx)  (or a fixed tau)
#   axis break (large gap):  no trace has data for dx > tau_axis
#                            tau_axis = n * max_j(tau_j)  (or a fixed one,
#                            never below max_j(tau_j))
#
# break_lines=False draws the line across every gap (no NaN rows);
# break_axis=False (or n = 0 with no fixed large gap) keeps one continuous
# axis. Either can be used alone: a break in the axis with the line drawn
# across it, or broken lines on a continuous axis, or both, or neither.
#
# geometric=True adds a second test for positive x: the step must also be
# large on a logarithmic scale,
#
#   ln(x_{i+1} / x_i) > K * median(ln(x_{j+1} / x_j)),
#
# so a log-spaced sweep (equal ratios, growing steps) has no gaps while a
# real hole in a linear or a log sweep is found on both scales. For time
# axes (values far from zero) the second test changes nothing and is left
# off; programs turn it on for non-time axes.


@dataclass(frozen=True)
class GapRule:
    """What a gap in a plotted series does (0.0.30); see the section above.

    Attributes
    ----------
    k:
        Small-gap multiplier K: a trace's line breaks where its step
        exceeds K x its median step (default GAP_K).
    n:
        Large-gap multiple n: the x axis breaks where no trace has data for
        longer than n x the largest small-gap threshold; 0 = never.
    manual:
        Fixed small gap in the units of x (> 0 replaces K x median).
    axis_manual:
        Fixed large gap in the units of x (> 0 replaces n; never below the
        small-gap threshold).
    break_lines:
        True: NaN rows at small gaps (the line stops and restarts).
    break_axis:
        True: Veusz axis-broken x axis at large gaps.
    geometric:
        True: a step must also be large on a log scale (positive x only).
    margin:
        Fraction of a gap kept visible at each side of an axis break.
    """
    k: float = GAP_K
    n: float = GAP_AXIS_N
    manual: float = 0.0
    axis_manual: float = 0.0
    break_lines: bool = True
    break_axis: bool = True
    geometric: bool = False
    margin: float = GAP_AXIS_MARGIN

    def line_threshold(self, t: Any) -> float:
        """Small-gap threshold of one trace in the units of ``t`` (0.0 = none)."""
        return gap_threshold(t, self.k, self.manual)

    def log_threshold(self, t: Any) -> float:
        """Small-gap threshold of ln(t) (0.0 when geometric is off, a fixed
        small gap is set, or some value is not positive)."""
        if not self.geometric or (self.manual and self.manual > 0.0):
            return 0.0
        values = np.asarray(t, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if values.size < 2 or np.any(values <= 0.0):
            return 0.0
        return gap_threshold(np.log(values), self.k, 0.0)

    def axis_threshold(self, line_threshold: float) -> float:
        """Large-gap threshold for an axis whose largest small-gap threshold
        is ``line_threshold``; 0.0 = no axis breaks."""
        if not self.break_axis:
            return 0.0
        if self.axis_manual and self.axis_manual > 0.0:
            return max(float(self.axis_manual), float(line_threshold))
        if self.n > 0.0 and line_threshold > 0.0:
            return max(float(self.n), 1.0) * float(line_threshold)
        return 0.0

    def axis_log_threshold(self, log_threshold: float) -> float:
        """Large-gap threshold of ln(x) (0.0 = no log condition)."""
        if (not self.break_axis or log_threshold <= 0.0
                or (self.axis_manual and self.axis_manual > 0.0) or self.n <= 0.0):
            return 0.0
        return max(float(self.n), 1.0) * float(log_threshold)

    def describe(self, unit: str = "") -> str:
        """One line for graph notes, every number exact (repr)."""
        def number(value: float) -> str:
            """``value`` with every digit (repr), without a trailing '.0'."""
            text = repr(float(value))
            return text[:-2] if text.endswith(".0") else text
        suffix = (" " + unit) if unit else ""
        scale = " (and on a log scale)" if self.geometric else ""
        if not self.break_lines:
            line = "lines drawn across gaps"
        elif self.manual and self.manual > 0.0:
            line = "line break where dx > %s%s" % (number(self.manual), suffix)
        else:
            line = "line break where dx > %s x median(dx)%s" % (number(self.k), scale)
        if not self.break_axis:
            axis = "no x-axis breaks"
        elif self.axis_manual and self.axis_manual > 0.0:
            axis = "x-axis break where no data for dx > %s%s" % (number(self.axis_manual), suffix)
        elif self.n > 0.0:
            axis = "x-axis break where no data for dx > %s x the small-gap threshold%s" % (
                number(self.n), scale)
        else:
            axis = "no x-axis breaks"
        return "Gaps: %s; %s." % (line, axis)


def rule_gap_positions(t: Any, rule: GapRule) -> np.ndarray:
    """Gap-row positions of one trace under ``rule`` (0.0.30): gap_positions()
    of its small-gap threshold, kept only where the step is also large on a
    log scale when the rule is geometric; empty when break_lines is off."""
    if not rule.break_lines:
        return np.zeros(0, dtype=np.int64)
    values = np.asarray(t, dtype=float).ravel()
    cut = gap_positions(values, rule.line_threshold(values))
    log_limit = rule.log_threshold(values)
    if cut.size and log_limit > 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_steps = np.log(values[cut]) - np.log(values[cut - 1])
        cut = cut[log_steps > log_limit]
    return cut


def rule_axis_break_pairs(series: Sequence[Any], rule: GapRule) -> list[tuple[float, float]]:
    """Axis-break intervals of one x axis shared by ``series`` (the x values
    of each trace) under ``rule`` (0.0.30); [] when break_axis is off."""
    arrays = [np.asarray(s, dtype=float).ravel() for s in series]
    arrays = [a for a in arrays if a.size]
    if not arrays or not rule.break_axis:
        return []
    line = max(rule.line_threshold(a) for a in arrays)
    logs = [rule.log_threshold(a) for a in arrays]
    log_limit = rule.axis_log_threshold(max(logs)) if all(v > 0.0 for v in logs) else 0.0
    return axis_break_pairs(np.concatenate(arrays), rule.axis_threshold(line), rule.margin,
                            log_threshold=log_limit)


def break_points(pairs: Sequence[tuple[float, float]]) -> list[float]:
    """Veusz axis-broken ``breakPoints``: the pairs as one flat list."""
    return [float(value) for pair in pairs for value in pair]


def gap_plan(x_series: Sequence[Any], rule: GapRule = GapRule()) -> tuple[list[np.ndarray], list[float]]:
    """Gap rows of every trace and the break points of their shared x axis
    (0.0.30), in one call.

    Returns (positions, x_breaks): ``positions[j]`` for insert_gaps() on the
    x and y of trace j (empty when its line stays whole), and ``x_breaks``
    for xy_page(x_breaks=) / Panel.x_breaks (empty = continuous axis).

        positions, x_breaks = gap_plan([t1, t2], GapRule(n=0))
        w.dataset_1d("t1_gaps", insert_gaps(t1, positions[0]))
        w.dataset_1d("v1_gaps", insert_gaps(v1, positions[0]))
    """
    positions = [rule_gap_positions(x, rule) for x in x_series]
    return positions, break_points(rule_axis_break_pairs(x_series, rule))


def local_veusz_exact(veusz_utc: Any, axis: LocalTimeAxis) -> np.ndarray:
    """Local wall-clock Veusz date values of UTC Veusz date values, as one
    addition: UTC + the offsets of ``axis`` (0.0.30).

    ``axis.veusz_seconds`` (local_time_from_veusz) goes through Unix epoch
    seconds (about 1.8e9 s, float64 steps of 2.4e-7 s) and back, which moves
    fractions of a second by up to about 1.2e-7 s; this sum keeps every
    digit Veusz date values hold (steps of 6e-8 s near 2026). Whole
    seconds are identical either way; NaN stays NaN.
    """
    return np.asarray(veusz_utc, dtype=float) + np.asarray(axis.offsets_s, dtype=float)
'''
once('''        pad = float(margin) * (stop - start)
        pairs.append((start + pad, stop - pad))
    return pairs


# %% ===========================================================================
#    SMITH CHARTS (0.0.26)''',
     '''        pad = float(margin) * (stop - start)
        pairs.append((start + pad, stop - pad))
    return pairs
''' + NEW_SECTION + '''

# %% ===========================================================================
#    SMITH CHARTS (0.0.26)''')

# ---- page helpers: x_breaks ---------------------------------------------
once('''def _add_axes(doc: VszDoc, xlabel: str, ylabel: str, xlog: bool, ylog: bool,
              xmin: Any, xmax: Any, ymin: Any, ymax: Any, xmode: str, escape: bool,
              x_autorange: str = "next-tick", y_autorange: str = "next-tick") -> None:
    """The x and y axes of the current graph, with labels, scales and limits.

    ``x_autorange``/``y_autorange`` are Veusz autoRange values used when the
    limit is "Auto": "next-tick" (Veusz default), "exact", "+5%", ...
    """
    doc.add("axis", "x")''',
     '''def _add_axes(doc: VszDoc, xlabel: str, ylabel: str, xlog: bool, ylog: bool,
              xmin: Any, xmax: Any, ymin: Any, ymax: Any, xmode: str, escape: bool,
              x_autorange: str = "next-tick", y_autorange: str = "next-tick",
              x_breaks: Sequence[float] = ()) -> None:
    """The x and y axes of the current graph, with labels, scales and limits.

    ``x_autorange``/``y_autorange`` are Veusz autoRange values used when the
    limit is "Auto": "next-tick" (Veusz default), "exact", "+5%", ...
    ``x_breaks`` (0.0.30): flat (start, stop, start, stop, ...) intervals to
    leave out; non-empty makes x a Veusz "axis-broken" axis.
    """
    breaks = [float(value) for value in x_breaks]
    if len(breaks) % 2:
        raise ValueError("x_breaks needs (start, stop) pairs; got %d values" % len(breaks))
    if breaks:
        doc.add("axis-broken", "x")
        doc.set("x/breakPoints", breaks)
    else:
        doc.add("axis", "x")''')
once('''            top_margin: str = "1.4cm", escape: bool = True,
            datetime_format: str = "", graph_name: str = "", notes: Any = "") -> str:
    """A page with one graph of line/marker traces; returns the graph name.''',
     '''            top_margin: str = "1.4cm", escape: bool = True,
            datetime_format: str = "", graph_name: str = "", notes: Any = "",
            x_breaks: Sequence[float] = ()) -> str:
    """A page with one graph of line/marker traces; returns the graph name.''')
once('''    notes:
        (0.0.25) Page notes, e.g. header_notes(source-file header).
    """
    graph = _graph_name(graph_name, page, traces)''',
     '''    notes:
        (0.0.25) Page notes, e.g. header_notes(source-file header).
    x_breaks:
        (0.0.30) Intervals left out of a broken x axis, flat (start, stop,
        ...) in x units, e.g. gap_plan()[1]; empty = one continuous axis.
    """
    graph = _graph_name(graph_name, page, traces)''')
once('''    _add_axes(doc, xlabel, ylabel, xlog, ylog, xmin, xmax, ymin, ymax, xmode, escape,
              x_autorange, y_autorange)
    if xmode == "datetime" and datetime_format:
        doc.set("x/TickLabels/format", datetime_format)
    _add_traces(doc, traces, escape)
    _add_key(doc, traces, key, key_position)''',
     '''    _add_axes(doc, xlabel, ylabel, xlog, ylog, xmin, xmax, ymin, ymax, xmode, escape,
              x_autorange, y_autorange, x_breaks=x_breaks)
    if xmode == "datetime" and datetime_format:
        doc.set("x/TickLabels/format", datetime_format)
    _add_traces(doc, traces, escape)
    _add_key(doc, traces, key, key_position)''')
once('''    datetime_format: str = ""                    # 0.0.24: datetime x tick format
    name: str = ""                               # 0.0.25: graph name ("g_<first y>")
''', '''    datetime_format: str = ""                    # 0.0.24: datetime x tick format
    name: str = ""                               # 0.0.25: graph name ("g_<first y>")
    x_breaks: Sequence[float] = ()               # 0.0.30: broken x axis (gap_plan)
''')
once('''        _add_axes(doc, panel.xlabel, panel.ylabel, panel.xlog, panel.ylog, panel.xmin, panel.xmax,
                  panel.ymin, panel.ymax, panel.xmode, escape, panel.x_autorange)''',
     '''        _add_axes(doc, panel.xlabel, panel.ylabel, panel.xlog, panel.ylog, panel.xmin, panel.xmax,
                  panel.ymin, panel.ymax, panel.xmode, escape, panel.x_autorange,
                  x_breaks=panel.x_breaks)''')
once('''def time_pair_pages(doc: VszDoc, page: str, traces: Sequence[Trace], mjd_x: str, local_x: str,
                    local_axis: LocalTimeAxis, title: str = "", ylabel: str = "",
                    mjd_label: str = "MJD (UTC)", **xy_kwargs: Any) -> list[str]:
    """Two xy pages from one set of traces: against MJD and against local time.

    ``mjd_x`` is a numeric dataset of MJD values; ``local_x`` a date dataset
    written from ``local_axis.veusz_seconds``. The trace x names are replaced
    per page. Returns ["<page>_mjd", "<page>_local"]. Extra keyword arguments
    go to xy_page() (xmode/datetime_format are set here).
    """
    for key in ("xmode", "datetime_format", "xlabel"):
        xy_kwargs.pop(key, None)
    xy_page(doc, page + "_mjd", with_x(traces, mjd_x), title=title, xlabel=mjd_label,
            ylabel=ylabel, **xy_kwargs)
    xy_kwargs.setdefault("x_autorange", "exact")
    xy_page(doc, page + "_local", with_x(traces, local_x), title=title, xlabel=local_axis.label,
            ylabel=ylabel, xmode="datetime", datetime_format=local_axis.datetime_format, **xy_kwargs)
    return [page + "_mjd", page + "_local"]''',
     '''def time_pair_pages(doc: VszDoc, page: str, traces: Sequence[Trace], mjd_x: str, local_x: str,
                    local_axis: LocalTimeAxis, title: str = "", ylabel: str = "",
                    mjd_label: str = "MJD (UTC)", mjd_breaks: Sequence[float] = (),
                    local_breaks: Sequence[float] = (), **xy_kwargs: Any) -> list[str]:
    """Two xy pages from one set of traces: against MJD and against local time.

    ``mjd_x`` is a numeric dataset of MJD values; ``local_x`` a date dataset
    written from ``local_axis.veusz_seconds``. The trace x names are replaced
    per page. Returns ["<page>_mjd", "<page>_local"]. Extra keyword arguments
    go to xy_page() (xmode/datetime_format are set here).
    (0.0.30) ``mjd_breaks`` / ``local_breaks``: x-axis breaks of each page
    in its own units (MJD days / Veusz date seconds), e.g. from gap_plan().
    """
    for key in ("xmode", "datetime_format", "xlabel", "x_breaks"):
        xy_kwargs.pop(key, None)
    xy_page(doc, page + "_mjd", with_x(traces, mjd_x), title=title, xlabel=mjd_label,
            ylabel=ylabel, x_breaks=mjd_breaks, **xy_kwargs)
    xy_kwargs.setdefault("x_autorange", "exact")
    xy_page(doc, page + "_local", with_x(traces, local_x), title=title, xlabel=local_axis.label,
            ylabel=ylabel, xmode="datetime", datetime_format=local_axis.datetime_format,
            x_breaks=local_breaks, **xy_kwargs)
    return [page + "_mjd", page + "_local"]''')

open(out_path, "w", encoding="utf-8", newline="").write(src)
print("wrote", out_path)
