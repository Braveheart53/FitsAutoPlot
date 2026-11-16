# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
docs/writer_0.0.30/add_entry_0030.py

Insert the 0.0.30 revision entry (with its own Function / Variable
Descriptions and More Info) into the writer made by make_0030.py, then
check that the result is the 0.0.30 master byte for byte.

Usage:  python add_entry_0030.py OUT.py

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
        (no functions; a script)
# %%%%% Variable Descriptions
    Define all utilized variables
        p: the writer file to edit
        s: its text
        a, b: start of the 0.0.29 and 0.0.28 entries
        sections: description sections copied from the 0.0.29 entry
        new_funcs: Function Description lines of the 0.0.30 additions
        entry: the 0.0.30 revision entry
        MASTER_0030: SHA-256 of vszh5_writer.py 0.0.30
# %%%%% More Info
    Run after make_0030.py (see its header).
=============================================================================
"""
# %% Imports
import hashlib
import sys

# %% Insert the 0.0.30 entry above the 0.0.29 entry
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
a = s.index("# %%%% 0.0.29:"); b = s.index("# %%%% 0.0.28:")
sections = s[a:b][s[a:b].index("# %%%%% Function Descriptions"):]
new_funcs = (
    "        GapRule (class): What a gap in a plotted series does (0.0.30); see the section above.\n"
    "        rule_gap_positions: Gap-row positions of one trace under ``rule`` (0.0.30): gap_positions() of its small-gap threshold, kept only where the step is also large on a log scale when the rule is geometric; empty when break_lines is off.\n"
    "        rule_axis_break_pairs: Axis-break intervals of one x axis shared by ``series`` (the x values of each trace) under ``rule`` (0.0.30); [] when break_axis is off.\n"
    "        break_points: Veusz axis-broken ``breakPoints``: the pairs as one flat list.\n"
    "        gap_plan: Gap rows of every trace and the break points of their shared x axis (0.0.30), in one call.\n"
    "        local_veusz_exact: Local wall-clock Veusz date values of UTC Veusz date values, as one addition: UTC + the offsets of ``axis`` (0.0.30).\n")
anchor = "        axis_break_pairs: (start, end) intervals to leave out of a broken axis.\n"
assert sections.count(anchor) == 1
sections = sections.replace(anchor, anchor + new_funcs)
sections = sections.replace("claude/vszh5-writer-0.0.29-notes.md (workbench docs/vszh5_writer_0.0.29_NOTES.md).",
                            "claude/vszh5-writer-0.0.30-notes.md (0.0.29: claude/vszh5-writer-0.0.29-notes.md).")
entry = """# %%%% 0.0.30: FitsAutoPlot suite 0.4.3 (operator request 2026-11-16: every plot
Date: 2026-11-16
#              generator gap aware, with options to break the line/trace
#              and/or the x axis), for every program that uses the writer
#              (MHM2010 / T4 Maser Timing Workbench included). Built on
#              master 0.0.29 (SHA-256 50631331...a504f9d701), registry
#              checked first. Additions only: every 0.0.29 call writes the
#              same file.
#              * NEW GapRule(k, n, manual, axis_manual, break_lines,
#                break_axis, geometric, margin): the 0.0.27 small / large
#                gap rule with either consequence optional - lines drawn
#                across gaps (break_lines=False), one continuous axis
#                (break_axis=False), both or neither - for any sorted x
#                (time, frequency, angle, index); geometric=True also
#                requires a step to be large on a log scale, so a
#                log-spaced sweep is not called gapped (10 MHz - 10 GHz,
#                201 points: 33 false line breaks with the linear rule,
#                none with it) while a hole in a linear or log sweep is
#                still found. describe() writes the rule with every digit.
#              * NEW rule_gap_positions, rule_axis_break_pairs,
#                break_points and gap_plan(x_series, rule): gap rows of
#                every trace and the break points of their shared x axis in
#                one call.
#              * NEW x_breaks= on xy_page() and Panel (grid_page), and
#                mjd_breaks= / local_breaks= on time_pair_pages(): a
#                non-empty list makes x a Veusz axis-broken axis with those
#                breakPoints; empty (the default) writes the 0.0.29 script.
#              * axis_break_pairs(log_threshold=0.0): optional log-scale
#                condition (default: the 0.0.27 result).
#              * NEW local_veusz_exact(veusz_utc, axis): local Veusz date
#                values as UTC + offset in one addition (the 2026-11-11
#                registry note: local_time_from_veusz().veusz_seconds go
#                through Unix seconds and move fractions by up to 1.2e-7 s).
#              Checked: FitsAutoPlot tests/test_writer_0030.py (documents of
#              every 0.0.29 helper identical object by object; the gap
#              options; Veusz 4.2.1 loads and renders broken axes from
#              xy_page / grid_page / time_pair_pages); Python 3.8, 3.12 and
#              3.13.
"""
s = s[:a] + entry + sections + s[a:]
open(p, "w", encoding="utf-8", newline="").write(s)
print("entry added")

# %% The result must be the 0.0.30 master (RF project registry) byte for byte
MASTER_0030 = "26194c8e96572fce2106bb5588d33d7c3f024fbd5169ff626cab2e40aa01f604"
digest = hashlib.sha256(open(p, "rb").read()).hexdigest()
assert digest == MASTER_0030, "result %s is not master 0.0.30" % digest
print("SHA-256 ok:", digest)
