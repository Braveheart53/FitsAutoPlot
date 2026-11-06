# -*- coding: utf-8 -*-
"""Native AutoPlot project model; files are written by vszh5_writer. No Qt or Veusz imports.

# %% Revision history
# %%%% 0.2.0: Files written through vszh5_writer 0.0.24 (the shared master).
# Date: 2026-11-05
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
# Date: 2026-10-11

The public SetData/Root.Add surface is deliberately small, so existing
AutoPlot page builders can target this model without a running Veusz process.
This is not a general-purpose replacement for the Veusz scripting API.
"""
from __future__ import annotations

import math
import os
from collections import OrderedDict

import numpy as np

import vszh5_writer
from vszh5_writer import VeuszWriter, VszDoc, vz_text

VERSION = "0.2.0"
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
    def __init__(self, widget, path):
        self._widget = widget
        self._path = path

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _Setting(self._widget, self._path + "/" + name)

    @property
    def val(self):
        return self._widget.settings.get(self._path)

    @val.setter
    def val(self, value):
        self._widget.settings[self._path] = _plain(value)   # fail before modifying


# %% In-memory widget tree
class NativeWidget:
    def __init__(self, document, kind, name, parent=None):
        self.document = document
        self.kind = kind
        self.name = name
        self.parent = parent
        self.children = []
        self.settings = OrderedDict()

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
        if self.parent is None:
            return "/"
        return self.parent.path.rstrip("/") + "/" + self.name

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        for child in self.children:
            if child.name == name:
                return child
        return _Setting(self, name)

    def Add(self, kind, name=None, autoadd=True, **settings):
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
        for key, value in settings.items():
            widget.settings[key] = _plain(value)
        self.children.append(widget)
        if kind == "graph" and autoadd:
            widget.Add("axis", name="x", direction="horizontal")
            widget.Add("axis", name="y", direction="vertical")
        return widget

    def Remove(self, name=None):
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
        if self.kind in ("axis", "axis-broken"):
            if self.settings.get("mode") == "datetime" and \
                    "TickLabels/size" not in self.settings:
                doc.set("TickLabels/size", "8pt")
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


# %% Dataset model and file output
class NativeDocument:
    """Build and atomically save a self-contained Veusz format-1 document."""

    def __init__(self, name="AutoPlot"):
        self.name = name
        self.Root = NativeWidget(self, "document", "")
        self.datasets = OrderedDict()
        self.tags = {}

    def _put(self, name, kind, parts):
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
        data = [v.decode("utf-8", "replace") if isinstance(v, bytes)
                else str(v) for v in values]
        if any("\x00" in v for v in data):
            raise ValueError("Text datasets cannot contain NUL")
        self._put(name, "text", {"data": data})

    def SetData2D(self, name, data, xrange=None, yrange=None,
                  xedge=None, yedge=None, xcent=None, ycent=None):
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
        self.SetData(name, seconds)
        self.datasets[name] = ("date", self.datasets[name][1])

    def TagDatasets(self, tag, names):
        if not tag or "/" in tag or "\x00" in tag or tag in (".", ".."):
            raise ValueError("Unsupported HDF5 tag name: %r" % tag)
        names = set(names)
        missing = names - set(self.datasets)
        if missing:
            raise KeyError("Unknown tagged datasets: %s" % sorted(missing))
        self.tags.setdefault(tag, set()).update(names)

    def GetDatasets(self):
        return list(self.datasets)

    def GetData(self, name):
        kind, parts = self.datasets[name]
        if kind in ("1d", "date"):
            return tuple(parts.get(k) for k in ("data", "serr", "nerr", "perr"))
        return (parts["data"],)

    def Get(self, path):
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
        if mode != "hdf5":
            raise ValueError("Only native HDF5 output is supported")
        return self.save(filename)

    def Close(self):
        """No external process to close; keep data alive until object disposal."""
