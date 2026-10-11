# -*- coding: utf-8 -*-
"""Native AutoPlot project model and HDF5 writer. No Qt or Veusz imports.

# %% Revision history
# %%%% 0.1.0: Direct .vszh5 serialization using h5py.
# Date: 2026-10-11

The public SetData/Root.Add surface is deliberately small, so existing
AutoPlot page builders can target this model without a running Veusz process.
This is not a general-purpose replacement for the Veusz scripting API.
"""
from __future__ import annotations

import datetime
import math
import os
import tempfile
from collections import OrderedDict

import h5py
import numpy as np

VERSION = "0.1.0"
DATASET_SETTINGS = frozenset((
    "xData", "yData", "labels", "scalePoints", "yDataLow", "yDataHigh",
    "xDataLow", "xDataHigh", "data",
))


# %% Safe script serialization
def _literal(value):
    """Generate Python literals only; never interpolate executable user text."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_literal(v) for v in value) + "]"
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite widget setting (dataset NaNs are supported)")
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError("Unsupported setting value: %s" % type(value).__name__)
    return repr(value)


def _escape_name(name):
    """Veusz HDF5 dataset-name escaping, not URL encoding."""
    return name.replace("`", "`BT").replace("/", "`SL")


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
        _literal(value)  # fail before modifying the document
        self._widget.settings[self._path] = value


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
        if kind not in ("page", "grid", "graph", "axis", "axis-broken",
                        "xy", "key", "image", "colorbar", "label"):
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
            _literal(value)
            widget.settings[key] = value
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

    def _write(self, lines):
        if self.kind == "graph" and "bottomMargin" not in self.settings:
            if any(child.kind in ("axis", "axis-broken") and
                   child.settings.get("mode") == "datetime"
                   for child in self.children):
                lines.append("Set('bottomMargin', '3cm')")
        if self.kind in ("axis", "axis-broken"):
            if self.settings.get("mode") == "datetime" and \
                    "TickLabels/size" not in self.settings:
                lines.append("Set('TickLabels/size', '8pt')")
        for key, value in self.settings.items():
            # AutoPlot captions are plain text, not Veusz math/markup.
            # Keep dataset names and dataset contents entirely unchanged.
            if key in ("label", "key") and isinstance(value, str):
                value = "".join("\\" + char if char in "_^{}" else char
                                for char in value)
            lines.append("Set(%r, %s)" % (key, _literal(value)))
        for child in self.children:
            lines.append("Add(%r, name=%r, autoadd=False)" %
                         (child.kind, child.name))
            lines.append("To(%r)" % child.name)
            child._write(lines)
            lines.append("To('..')")


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

    def SetData(self, name, data, symerr=None, negerr=None, poserr=None):
        parts = {}
        for key, values in (("data", data), ("serr", symerr),
                            ("nerr", negerr), ("perr", poserr)):
            if values is not None:
                array = np.array(values, dtype=np.float64, copy=True)
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
        if kind == "1d":
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
        lines = ["# AutoPlot native project writer %s" % VERSION,
                 "# Veusz format-1; compatible settings baseline 3.4.",
                 "SetCompatLevel(0)"]
        self.Root._write(lines)
        return "\n".join(lines) + "\n"

    GetSaveString = SaveToString

    def save(self, filename, minimized=False):
        """Write a same-directory temporary file, then replace only on success."""
        filename = os.fspath(filename)
        if not filename.lower().endswith(".vszh5"):
            filename = os.path.splitext(filename)[0] + ".vszh5"
        used = self.referenced_datasets()  # validate even a full save
        names = used if minimized else set(self.datasets)
        script = self.SaveToString().encode("utf-8")
        folder = os.path.dirname(os.path.abspath(filename))
        fd, temporary = tempfile.mkstemp(prefix=".autoplot-", suffix=".vszh5",
                                         dir=folder)
        os.close(fd)
        try:
            with h5py.File(temporary, "w") as h:
                root = h.create_group("Veusz")
                # This is a compatibility baseline, not a claimed runtime.
                root.attrs["vsz_version"] = "3.4"
                root.attrs["vsz_format"] = 1
                root.attrs["vsz_saved_at"] = datetime.datetime.now(
                    datetime.timezone.utc).replace(tzinfo=None).isoformat()
                root.attrs["autoplot_writer_version"] = VERSION
                data_group = root.create_group("Data")
                document = root.create_group("Document")
                tags = document.create_group("Tags")
                document.create_dataset("document", data=[script])
                utf8 = h5py.string_dtype(encoding="utf-8")
                for name in sorted(names):
                    kind, parts = self.datasets[name]
                    group = data_group.create_group(_escape_name(name))
                    group.attrs["vsz_datatype"] = kind
                    for part, values in parts.items():
                        if kind == "text":
                            dataset = group.create_dataset(part, data=values,
                                                           dtype=utf8)
                        else:
                            kwargs = {}
                            if np.asarray(values).size:
                                kwargs = {"compression": "gzip",
                                          "compression_opts": 4, "shuffle": True}
                            dataset = group.create_dataset(part, data=values, **kwargs)
                        suffix = {"serr": " (+-)", "nerr": " (-)",
                                  "perr": " (+)"}.get(part, "")
                        if part in ("data", "serr", "nerr", "perr"):
                            dataset.attrs["vsz_name"] = name + suffix
                            if kind == "date" and part == "data":
                                dataset.attrs["vsz_convert_datetime"] = 1
                        elif kind == "2d":
                            group["data"].attrs["vsz_" + part] = dataset.ref
                for tag, members in sorted(self.tags.items()):
                    selected = sorted(members & names)
                    if selected:
                        tags.create_dataset(tag, data=selected, dtype=utf8)
                h.flush()
            os.replace(temporary, filename)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return filename

    def Save(self, filename, mode="hdf5"):
        if mode != "hdf5":
            raise ValueError("Only native HDF5 output is supported")
        return self.save(filename)

    def Close(self):
        """No external process to close; keep data alive until object disposal."""
