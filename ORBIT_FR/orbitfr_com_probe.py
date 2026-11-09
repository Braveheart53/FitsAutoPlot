# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
orbitfr_com_probe.py

Read-only discovery of the COM automation interface that MVG ORBIT/FR
959 Spectrum (and related ORBIT/FR software) registers on a Windows PC.

MVG documents that 959 Spectrum scripting is "based on MS COM technology"
and that "all scripting environments that support COM can be used", but the
object model (ProgIDs, objects, methods) ships only with the installed
software.  This script finds what is registered and writes the type-library
contents (objects, methods, properties, arguments, enumerations) to a
Markdown report, so ORBITFR_AutoPlot.py can be written against the real API.

It NEVER creates a COM object, so it cannot start 959 Spectrum, move a
positioner or touch an instrument.  It only reads the registry and type
libraries.

Usage (on the measurement PC, Windows, Python with pywin32):
    python orbitfr_com_probe.py                    # find candidates, dump them
    python orbitfr_com_probe.py --pattern "959|orbit|mvg"
    python orbitfr_com_probe.py --typelib-file "C:\\Program Files (x86)\\...\\x.tlb"
    python orbitfr_com_probe.py --list-only

Output: orbitfr_com_report_<YYYYmmdd_HHMMSS>.md in the current folder.

Created on 2026-11-05

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216
Company: NRAO / Green Bank Observatory (GBO)
Licence: GPL-3.0-or-later

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.2: Header template (no functional change).
Date: 2026-11-09
#              * Header in the author's template: author contact lines and,
#                under this entry, Function Descriptions, Variable
#                Descriptions and More Info.
#              * Docstrings for every function, class and method.  No change
#                to what the program does.
# %%%%% Function Descriptions
        _read_default: Default value of a registry key, or None.
        _subkeys: Names of the subkeys of a registry key.
        find_progids: ProgIDs (HKCR\\<ProgID>\\CLSID) whose name matches ``pattern``.
        find_typelibs: Registered type libraries (HKCR\\TypeLib) whose title matches ``pattern``.
        dump_typelib: Markdown lines describing every type in a loaded ITypeLib.
        load_registered_typelib: pythoncom.LoadRegTypeLib for 'major.minor' (hex) version strings.
        main: Scan, dump, write the report; returns a process exit code.
# %%%%% Variable Descriptions
    Define all utilized variables
        DEFAULT_PATTERN: regular expression of ORBIT/FR names searched for in the registry
# %%%%% More Info
    Windows only (pywin32): lists the ORBIT/FR COM registrations and dumps the
    type library for the planned ORBIT/FR AutoPlot.  See
    ORBIT_FR/ORBIT_FR_API_REQUIREMENTS.md.
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.0.1: First internal draft - registry scan and type-library dump
Date: 2026-11-05
    Scans HKCR ProgIDs and HKLM\\SOFTWARE\\Classes (64- and 32-bit views)
    for name patterns; dumps registered type libraries with pythoncom
    (LoadRegTypeLib / LoadTypeLib).  No object instantiation.
=============================================================================
"""
# %% Import Modules
from __future__ import annotations

import argparse
import datetime
import platform
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__version__ = "0.0.2"

DEFAULT_PATTERN = r"959|orbit|ofr|mvg|datapro|midas|spectrum|fr959|al-?4"

if sys.platform != "win32":  # pragma: no cover - only meaningful on Windows
    winreg = None
else:
    import winreg


# %% Registry scan
def _read_default(root, path: str, access: int = 0) -> Optional[str]:
    """Default value of a registry key, or None."""
    try:
        with winreg.OpenKey(root, path, 0, winreg.KEY_READ | access) as key:
            value, _kind = winreg.QueryValueEx(key, "")
            return str(value)
    except OSError:
        return None


def _subkeys(root, path: str, access: int = 0) -> List[str]:
    """Names of the subkeys of a registry key."""
    names = []
    try:
        with winreg.OpenKey(root, path, 0, winreg.KEY_READ | access) as key:
            index = 0
            while True:
                try:
                    names.append(winreg.EnumKey(key, index))
                except OSError:
                    break
                index += 1
    except OSError:
        pass
    return names


def find_progids(pattern: re.Pattern) -> List[Dict[str, str]]:
    """
    ProgIDs (HKCR\\<ProgID>\\CLSID) whose name matches ``pattern``.

    Both registry views (64-bit and 32-bit) are searched, because a 32-bit
    959 Spectrum registers under WOW6432Node.
    """
    found: Dict[Tuple[str, str], Dict[str, str]] = {}
    views = [("64-bit", winreg.KEY_WOW64_64KEY), ("32-bit", winreg.KEY_WOW64_32KEY)]
    for view_name, flag in views:
        base = r"SOFTWARE\Classes"
        for name in _subkeys(winreg.HKEY_LOCAL_MACHINE, base, flag):
            if not pattern.search(name):
                continue
            clsid = _read_default(winreg.HKEY_LOCAL_MACHINE,
                                  base + "\\" + name + r"\CLSID", flag)
            if not clsid:
                continue
            cls_path = base + "\\CLSID\\" + clsid
            entry = {
                "progid": name, "clsid": clsid, "view": view_name,
                "description": _read_default(winreg.HKEY_LOCAL_MACHINE, cls_path, flag) or "",
                "inproc": _read_default(winreg.HKEY_LOCAL_MACHINE,
                                        cls_path + r"\InprocServer32", flag) or "",
                "local": _read_default(winreg.HKEY_LOCAL_MACHINE,
                                       cls_path + r"\LocalServer32", flag) or "",
                "typelib": _read_default(winreg.HKEY_LOCAL_MACHINE,
                                         cls_path + r"\TypeLib", flag) or "",
            }
            found[(name, view_name)] = entry
    return sorted(found.values(), key=lambda e: (e["progid"], e["view"]))


def find_typelibs(pattern: re.Pattern) -> List[Dict[str, str]]:
    """Registered type libraries (HKCR\\TypeLib) whose title matches ``pattern``."""
    libs = []
    for guid in _subkeys(winreg.HKEY_CLASSES_ROOT, "TypeLib"):
        for version in _subkeys(winreg.HKEY_CLASSES_ROOT, "TypeLib\\" + guid):
            title = _read_default(winreg.HKEY_CLASSES_ROOT,
                                  "TypeLib\\%s\\%s" % (guid, version)) or ""
            if pattern.search(title):
                path = ""
                for arch in ("win64", "win32"):
                    path = _read_default(winreg.HKEY_CLASSES_ROOT,
                                         "TypeLib\\%s\\%s\\0\\%s" % (guid, version, arch)) or path
                libs.append({"guid": guid, "version": version, "title": title,
                             "path": path})
    return libs


# %% Type-library dump (pythoncom; nothing is instantiated)
def dump_typelib(typelib) -> List[str]:
    """
    Markdown lines describing every type in a loaded ITypeLib.

    Coclasses, dispatch / dual interfaces (methods, property get/put and
    argument names), enumerations with values, records.
    """
    import pythoncom

    kinds = {pythoncom.TKIND_ENUM: "enum", pythoncom.TKIND_RECORD: "record",
             pythoncom.TKIND_MODULE: "module", pythoncom.TKIND_INTERFACE: "interface",
             pythoncom.TKIND_DISPATCH: "dispatch", pythoncom.TKIND_COCLASS: "coclass",
             pythoncom.TKIND_ALIAS: "alias", pythoncom.TKIND_UNION: "union"}
    invkinds = {pythoncom.INVOKE_FUNC: "method",
                pythoncom.INVOKE_PROPERTYGET: "get",
                pythoncom.INVOKE_PROPERTYPUT: "put",
                pythoncom.INVOKE_PROPERTYPUTREF: "putref"}
    lines = []
    name, doc, _ctx, _file = typelib.GetDocumentation(-1)
    lines.append("### Type library `%s`" % name)
    if doc:
        lines.append("")
        lines.append(doc)
    for index in range(typelib.GetTypeInfoCount()):
        info = typelib.GetTypeInfo(index)
        attr = info.GetTypeAttr()
        tname, tdoc, _c, _f = typelib.GetDocumentation(index)
        lines.append("")
        lines.append("#### %s `%s`  (GUID %s)" % (kinds.get(attr.typekind, attr.typekind),
                                                  tname, attr.iid))
        if tdoc:
            lines.append("> %s" % tdoc)
        if attr.typekind == pythoncom.TKIND_COCLASS:
            for k in range(attr.cImplTypes):
                ref = info.GetRefTypeInfo(info.GetRefTypeOfImplType(k))
                lines.append("- implements `%s`" % ref.GetDocumentation(-1)[0])
        for f in range(attr.cFuncs):
            fdesc = info.GetFuncDesc(f)
            names = info.GetNames(fdesc.memid)
            fdoc = info.GetDocumentation(fdesc.memid)[1] or ""
            lines.append("- %s `%s(%s)`%s" % (
                invkinds.get(fdesc.invkind, fdesc.invkind), names[0],
                ", ".join(names[1:]), (" - " + fdoc) if fdoc else ""))
        for v in range(attr.cVars):
            vdesc = info.GetVarDesc(v)
            vname = info.GetNames(vdesc.memid)[0]
            value = getattr(vdesc, "value", None)
            lines.append("- `%s`%s" % (vname, (" = %r" % (value,)) if value is not None else ""))
    return lines


def load_registered_typelib(guid: str, version: str):
    """pythoncom.LoadRegTypeLib for 'major.minor' (hex) version strings."""
    import pythoncom
    major, _, minor = version.partition(".")
    return pythoncom.LoadRegTypeLib(guid, int(major, 16), int(minor or "0", 16), 0)


# %% Main
def main(argv: Optional[List[str]] = None) -> int:
    """Scan, dump, write the report; returns a process exit code."""
    parser = argparse.ArgumentParser(
        description="Read-only discovery of ORBIT/FR (MVG 959 Spectrum) COM automation objects.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help="case-insensitive regex for ProgID / type-library names")
    parser.add_argument("--typelib-file", action="append", default=[],
                        help="also dump this .tlb/.dll/.exe/.olb (repeatable)")
    parser.add_argument("--list-only", action="store_true",
                        help="list candidates without dumping type libraries")
    parser.add_argument("--out", default="",
                        help="report file (default orbitfr_com_report_<time>.md)")
    args = parser.parse_args(argv)

    if winreg is None:
        print("This probe reads the Windows registry; run it on the 959 Spectrum PC.")
        return 2
    pattern = re.compile(args.pattern, re.IGNORECASE)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out or "orbitfr_com_report_%s.md" % stamp)

    lines = ["# ORBIT/FR COM discovery report", "",
             "| Item | Value |", "|---|---|",
             "| Probe version | %s |" % __version__,
             "| Date | %s |" % datetime.datetime.now().isoformat(timespec="seconds"),
             "| Computer | %s |" % platform.node(),
             "| Windows | %s |" % platform.platform(),
             "| Python | %s (%s) |" % (platform.python_version(), platform.architecture()[0]),
             "| Pattern | `%s` |" % args.pattern.replace("|", "\\|"), ""]

    progids = find_progids(pattern)
    lines += ["## Registered ProgIDs", ""]
    if progids:
        lines += ["| ProgID | View | Description | Server | TypeLib |", "|---|---|---|---|---|"]
        for e in progids:
            server = e["local"] or e["inproc"]
            lines.append("| `%s` | %s | %s | `%s` | %s |" % (
                e["progid"], e["view"], e["description"], server, e["typelib"]))
    else:
        lines.append("None matched. Try a wider `--pattern`, or look in OLE/COM "
                     "Object Viewer (oleview.exe) under 'Automation Objects'.")
    libs = find_typelibs(pattern)
    lines += ["", "## Registered type libraries", ""]
    if libs:
        lines += ["| Title | GUID | Version | File |", "|---|---|---|---|"]
        for lib in libs:
            lines.append("| %s | %s | %s | `%s` |" % (lib["title"], lib["guid"],
                                                       lib["version"], lib["path"]))
    else:
        lines.append("None matched.")

    if not args.list_only:
        try:
            import pythoncom
        except ImportError:
            lines += ["", "pywin32 is not installed (`mamba install pywin32` or "
                      "`pip install pywin32`); type libraries were not dumped."]
            pythoncom = None
        if pythoncom is not None:
            lines += ["", "## Type library contents", ""]
            seen = set()
            targets = [("reg", lib["guid"], lib["version"]) for lib in libs]
            targets += [("reg", e["typelib"], "") for e in progids if e["typelib"]]
            targets += [("file", path, "") for path in args.typelib_file]
            for kind, ident, version in targets:
                if (kind, ident, version) in seen:
                    continue
                seen.add((kind, ident, version))
                try:
                    if kind == "file":
                        tlb = pythoncom.LoadTypeLib(ident)
                    elif version:
                        tlb = load_registered_typelib(ident, version)
                    else:
                        versions = _subkeys(winreg.HKEY_CLASSES_ROOT, "TypeLib\\" + ident)
                        if not versions:
                            continue
                        tlb = load_registered_typelib(ident, versions[-1])
                    lines += dump_typelib(tlb) + [""]
                except Exception as exc:  # report and continue
                    lines.append("- could not load `%s` %s: %s" % (ident, version, exc))

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote %s (%d ProgIDs, %d type libraries matched)" % (out, len(progids), len(libs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
