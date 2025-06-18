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


def fastest_file_parser(filename, line_targets=None, string_patterns=None):
    """
    Comprehensive fast file parser for ASCII text files.

    Args:
        filename: Path to ASCII text file
        line_targets: List of specific line numbers to extract
        string_patterns: Dict of string patterns to search for

    Returns:
        Dictionary containing extracted data
    """
    import os

    extracted_data = {
        'metadata': {'total_lines': 0, 'file_size': 0},
        'line_data': {},
        'pattern_matches': {},
        'user_variables': {}
    }

    # import os
    extracted_data['metadata']['file_size'] = os.path.getsize(filename)

    # Use appropriate method based on file size
    if extracted_data['metadata']['file_size'] < 10**7:  # < 10MB
        # Use fast binary reading
        with open(filename, 'rb') as file:
            content = file.read().decode('ascii')
            lines = content.splitlines()
    else:
        # Use memory-efficient line iteration
        with open(filename, 'r', encoding='ascii') as file:
            lines = file.readlines()

    # Process lines
    for line_number, line in enumerate(lines, 1):
        clean_line = line.strip()
        extracted_data['metadata']['total_lines'] = line_number

        # Extract specific line numbers
        if line_targets and line_number in line_targets:
            extracted_data['line_data'][line_number] = clean_line

        # Search for string patterns
        if string_patterns:
            for pattern_name, pattern in string_patterns.items():
                if pattern in clean_line:
                    extracted_data['pattern_matches'][pattern_name] = {
                        'line_number': line_number,
                        'content': clean_line,
                        'extracted_value': clean_line.split(
                            pattern,
                            1)[1].strip() if ':' in pattern else clean_line
                    }

    return extracted_data
