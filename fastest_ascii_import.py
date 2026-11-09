# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------

Created on 2025-06-16

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216


# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.4: Header template, docstrings, __version__ (no functional change).
Date: 2026-11-09
#              * Header in the author's template: author contact lines and,
#                under this entry, Function Descriptions, Variable
#                Descriptions and More Info.
#              * Docstrings for every function, class and method.  No change
#                to what the program does.
# %%%%% Function Descriptions
        _data_values: Floats of a comma-separated line of numbers, else [] (strict).
        _after: Text after the first case-insensitive occurrence of ``pattern``.
        fastest_file_parser: Comprehensive fast file parser for ASCII text files.
# %%%%% Variable Descriptions
    Define all utilized variables
        _NUMBER: a line holding one number
        SFT_FIRST_DATA_LINE: .sft header occupies lines 1-57
# %%%%% More Info
    Kept for other scripts; the R&S plotter has its own .sft reader.
    Requirements, optional packages and the tested Python 3.8 / 3.12
    environments: REQUIREMENTS.md (mamba: environment-py38.yml,
    environment-py312.yml).  Release notes: RELEASE_suite-2026-11-09.md.
# %%%% 0.0.3: Robust decoding and data-line detection.
Date: 2026-11-05
# %%%%% Function Descriptions
        fastest_file_parser: same interface and result layout.
        * Files are decoded as ASCII with replacement characters (a single
          byte above 127 used to raise UnicodeDecodeError) and read in one
          pass for every size (the > 10 MB branch read the file twice as
          large into a list anyway).
        * Data lines: every comma-separated field must be a number.  The
          deprecated np.fromstring(sep=',') accepted the leading numbers of
          a line ("801;" -> [801.0]) and warned on every header line.
        * Pattern values: the text after the pattern is cut case-
          insensitively, like the match itself (an upper-case pattern on a
          mixed-case line raised IndexError).
        * The ".sft only from line 58" rule also applies to ".SFT".
        * History is newest-first, as in the rest of the repository.
        (RAndS_FSW_ASCII_Plotter 1.1.0 has its own .sft reader; this module
        stays for other scripts.)
# %%%% 0.0.2: Start .sft data at line 58 to avoid data size mismatches.
Date: 2025-09-02
# %%%% 0.0.1: Script to run in consol description
Date: 2025-06-16
# %%%%% Function Descriptions
        fastest_file_parser: parse any ASCII file per the commands passed


# %%%%% Variable Descriptions
    Define all utilized variables
        file_path: path(s) to selected files for processing

# %%%%% More Info
# %%%%%% Usage example
# =============================================================================
# result = fastest_file_parser(
#     'data.txt',
#     line_targets=[1, 5, 10],
#     string_patterns={'header': 'HEADER:', 'data': 'DATA:', 'config': 'CONFIG:'}
# )
#
# # Example 1: Process only specific lines with pattern matching
# result = fastest_file_parser(
#     'data.txt',
#     line_targets=[1, 5, 10],
#     string_patterns={'header': 'HEADER:', 'data': 'DATA:', 'config': 'CONFIG:'}
# )
#
# print("Specific lines processing:")
# print(f"Lines extracted: {list(result['line_data'].keys())}")
# print(f"Pattern matches found: {len(result['pattern_matches'])}")
#
# # Example 2: Process ALL lines with pattern matching (line_targets=None)
# result = fastest_file_parser(
#     'data.txt',
#     line_targets=None,  # Explicitly passing None to process all lines
#     string_patterns={'header': 'HEADER:', 'data': 'DATA:', 'config': 'CONFIG:'}
# )
#
# print("\nAll lines processing:")
# print(f"Total lines processed: {result['metadata']['total_lines']}")
# print(f"All lines extracted: {len(result['line_data'])}")
# print(
#     f"Pattern matches found: {sum(len(matches) for matches in result['pattern_matches'].values())}")
=============================================================================
"""


# %% Function Definitions


import re as _re

__version__ = "0.0.4"

_NUMBER = _re.compile(r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$")
SFT_FIRST_DATA_LINE = 58          # .sft header occupies lines 1-57


def _data_values(line):
    """Floats of a comma-separated line of numbers, else [] (strict)."""
    fields = line.strip().rstrip(",").split(",")
    if not fields or not all(_NUMBER.match(f) for f in fields):
        return []
    return [float(f) for f in fields]


def _after(pattern, line):
    """Text after the first case-insensitive occurrence of ``pattern``."""
    parts = _re.split(_re.escape(pattern), line, maxsplit=1, flags=_re.IGNORECASE)
    return parts[1].strip() if len(parts) > 1 else ""


def fastest_file_parser(filename, line_targets=None, string_patterns=None):
    """
    Comprehensive fast file parser for ASCII text files.

    Args:
        filename: Path to ASCII text file
        line_targets: List of specific line numbers to extract (1-based)
        string_patterns: Dict of string patterns to search for
            (case-insensitive; every occurrence is kept, the 2nd as
            "<name>_2", ...)

    Returns:
        Dictionary with 'metadata', 'line_data', 'pattern_matches',
        'data_matches' ('data', 'data_2', ... : line_number, content,
        extracted_value list of floats) and 'user_variables'.
    """
    import os
    from collections import defaultdict

    extracted_data = {
        'metadata': {'total_lines': 0, 'file_size': os.path.getsize(filename)},
        'line_data': {},
        'pattern_matches': {},
        'data_matches': {},
        'user_variables': {}
    }
    with open(filename, 'rb') as handle:
        lines = handle.read().decode('ascii', errors='replace').splitlines()
    targets = set(line_targets or ())
    is_sft = os.path.splitext(filename)[1].lower() == ".sft"
    patterns = [(name, pattern, pattern.casefold())
                for name, pattern in (string_patterns or {}).items()]
    pattern_tracking = defaultdict(int)
    for line_number, line in enumerate(lines, 1):
        clean_line = line.strip()
        extracted_data['metadata']['total_lines'] = line_number
        if line_number in targets:
            extracted_data['line_data'][line_number] = clean_line
        folded = clean_line.casefold()
        for pattern_name, pattern, folded_pattern in patterns:
            if folded_pattern in folded:
                pattern_tracking[pattern_name] += 1
                count = pattern_tracking[pattern_name]
                key = pattern_name if count == 1 else '%s_%d' % (pattern_name, count)
                extracted_data['pattern_matches'][key] = {
                    'line_number': line_number,
                    'content': clean_line,
                    'extracted_value': (_after(pattern, clean_line) if ':' in pattern
                                        else clean_line),
                }
        values = _data_values(clean_line)
        if values and not (is_sft and line_number < SFT_FIRST_DATA_LINE):
            pattern_tracking['data_located'] += 1
            count = pattern_tracking['data_located']
            key = 'data' if count == 1 else 'data_%d' % count
            extracted_data['data_matches'][key] = {
                'line_number': line_number,
                'content': clean_line,
                'extracted_value': values,
            }
    return extracted_data
