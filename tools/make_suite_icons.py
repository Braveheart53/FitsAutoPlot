# -*- coding: utf-8 -*-
"""
=============================================================================
# %% Header Info
--------
tools/make_suite_icons.py

Draw the launcher's icons into assets/ (256 x 256 px PNG, transparent
corners): one per program button plus the application icon, so the suite
ships real images instead of *.placeholder notes.

    python tools/make_suite_icons.py            # writes into ../assets
    python tools/make_suite_icons.py OUT_DIR    # writes elsewhere

The application icon (app_icon.png) is a square crop of the user's own
photograph assets/GBT_2.ico (the Green Bank Telescope at night); the program
icons are simple drawings of each program's data (antenna pattern, time
series, timing pulses, table, spectrum, Smith chart, antenna range) with the
file type as text.  No third-party logos or brand names are used.

Created on 2026-11-09

# %%% Author Information
@author: William W. Wallace
Author Email: wwallace@nrao.edu
Author Secondary Email: naval.antennas@gmail.com
Author Business Phone: +1 (304) 456-2216

# %%% Revisions
--------
Utilizing Semantic Schema as External Release.Internal Release.Working version

# %%%% 0.0.1: First version - eight icons for the launcher 0.2.1.
Date: 2026-11-09
# %%%%% Function Descriptions
        tile: Rounded coloured square on a transparent 256 px canvas; returns (figure, axes).
        finish: Write the label at the bottom of the tile and save the figure as a PNG.
        _polar: Plot r(theta) (r in icon units) about (cx, cy); theta 0 points up.
        draw_atr: Antenna pattern: polar grid and a main lobe with side lobes.
        draw_fits: Time series with a gap (the line breaks), as on the FITS pages.
        draw_franks: Timing data: one-pulse-per-second ticks over a slow drift.
        draw_csv: Table: header row and a 3 x 4 grid of cells.
        draw_sft: Spectrum analyser trace: noise floor with two carriers.
        draw_snp: Smith chart: unit circle, resistance circles and reactance arcs.
        draw_range: Antenna range: a dish facing a probe across the range.
        make_app_icon: Square crop of GBT_2.ico (the telescope), resized to 256 px.
        main: Write every icon; returns the list of files written.
# %%%%% Variable Descriptions
    Define all utilized variables
        SIZE_PX = 256
        COLOURS = { "atr": "#0f766e", # teal "fits": "#3730a3", # indigo "franks": "#...
        WHITE: ffffff"
        ICONS = { "atr_icon.png": draw_atr, "fits_icon.png": draw_fits, "franks_ico...
# %%%%% More Info
    See README.md and REQUIREMENTS.md.
=============================================================================
"""
# %% Imports
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                    # no display needed
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle   # noqa: E402

__version__ = "0.0.1"

# %% Settings
SIZE_PX = 256
COLOURS = {
    "atr": "#0f766e",      # teal
    "fits": "#3730a3",     # indigo
    "franks": "#9a3412",   # burnt orange
    "csv": "#166534",      # green
    "sft": "#991b1b",      # red
    "snp": "#1e40af",      # blue
    "range": "#334155",    # slate
}
WHITE = "#ffffff"


# %% Drawing helpers
def tile(colour: str):
    """Rounded coloured square on a transparent 256 px canvas; returns (figure, axes).

    The axes span the whole figure with data limits 0..1 in both directions,
    so glyphs are drawn in fractions of the icon.
    """
    fig = plt.figure(figsize=(SIZE_PX / 100.0, SIZE_PX / 100.0), dpi=100)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.04, 0.04), 0.92, 0.92,
                                boxstyle="round,pad=0,rounding_size=0.16",
                                facecolor=colour, edgecolor="none"))
    return fig, ax


def finish(fig, ax, label: str, path: Path) -> Path:
    """Write the label at the bottom of the tile and save the figure as a PNG."""
    ax.text(0.5, 0.15, label, color=WHITE, ha="center", va="center",
            fontsize=30 if len(label) <= 4 else 24, fontweight="bold",
            family="DejaVu Sans")
    fig.savefig(str(path), dpi=100, transparent=True)
    plt.close(fig)
    return path


def _polar(ax, cx, cy, r, theta, **kw):
    """Plot r(theta) (r in icon units) about (cx, cy); theta 0 points up."""
    ax.plot(cx + r * np.sin(theta), cy + r * np.cos(theta), **kw)


# %% The icons
def draw_atr(path: Path) -> Path:
    """Antenna pattern: polar grid and a main lobe with side lobes."""
    fig, ax = tile(COLOURS["atr"])
    cx, cy, rmax = 0.5, 0.58, 0.30
    for frac in (0.33, 0.66, 1.0):
        ax.add_patch(Circle((cx, cy), rmax * frac, fill=False, ec=WHITE, lw=1.2, alpha=0.45))
    theta = np.linspace(-np.pi, np.pi, 721)
    u = 6.0 * np.sin(theta / 1.0)
    pattern = np.abs(np.sinc(u / np.pi)) ** 0.6            # main lobe up, side lobes
    pattern = np.where(np.abs(theta) < np.pi / 2, pattern, 0.12 * pattern)
    _polar(ax, cx, cy, rmax * pattern, theta, color=WHITE, lw=3.2)
    return finish(fig, ax, "ATR", path)


def draw_fits(path: Path) -> Path:
    """Time series with a gap (the line breaks), as on the FITS pages."""
    fig, ax = tile(COLOURS["fits"])
    t1 = np.linspace(0.14, 0.46, 60)
    t2 = np.linspace(0.58, 0.86, 50)
    y1 = 0.55 + 0.10 * np.sin(18 * t1) + 0.03 * np.sin(55 * t1)
    y2 = 0.60 + 0.09 * np.sin(18 * t2 + 1.0) + 0.03 * np.sin(61 * t2)
    ax.plot([0.12, 0.88], [0.36, 0.36], color=WHITE, lw=1.5, alpha=0.6)   # time axis
    ax.plot(t1, y1, color=WHITE, lw=3.0)
    ax.plot(t2, y2, color=WHITE, lw=3.0)
    return finish(fig, ax, "FITS", path)


def draw_franks(path: Path) -> Path:
    """Timing data: one-pulse-per-second ticks over a slow drift."""
    fig, ax = tile(COLOURS["franks"])
    x = np.linspace(0.14, 0.86, 300)
    drift = 0.52 + 0.12 * (x - 0.14) / 0.72 + 0.03 * np.sin(20 * x)
    ax.plot(x, drift, color=WHITE, lw=3.0)
    for xp in np.linspace(0.18, 0.82, 7):
        ax.plot([xp, xp], [0.36, 0.44], color=WHITE, lw=2.4, alpha=0.8)
    return finish(fig, ax, "MJD", path)


def draw_csv(path: Path) -> Path:
    """Table: header row and a 3 x 4 grid of cells."""
    fig, ax = tile(COLOURS["csv"])
    x0, y0, w, h = 0.18, 0.34, 0.64, 0.46
    ax.add_patch(Rectangle((x0, y0 + h * 0.75), w, h * 0.25, color=WHITE, alpha=0.9))
    for i in range(5):
        ax.plot([x0, x0 + w], [y0 + h * i / 4] * 2, color=WHITE, lw=2.0)
    for j in range(4):
        ax.plot([x0 + w * j / 3] * 2, [y0, y0 + h], color=WHITE, lw=2.0)
    return finish(fig, ax, "CSV", path)


def draw_sft(path: Path) -> Path:
    """Spectrum analyser trace: noise floor with two carriers."""
    fig, ax = tile(COLOURS["sft"])
    rng = np.random.default_rng(7)
    f = np.linspace(0.14, 0.86, 240)
    trace = 0.42 + 0.025 * rng.standard_normal(f.size)
    for fc, a in ((0.38, 0.36), (0.66, 0.24)):
        trace = trace + a * np.exp(-((f - fc) / 0.012) ** 2)
    ax.plot(f, trace, color=WHITE, lw=2.2)
    return finish(fig, ax, "SFT", path)


def draw_snp(path: Path) -> Path:
    """Smith chart: unit circle, resistance circles and reactance arcs."""
    fig, ax = tile(COLOURS["snp"])
    cx, cy, R = 0.5, 0.58, 0.30
    ax.add_patch(Circle((cx, cy), R, fill=False, ec=WHITE, lw=2.6))
    ax.plot([cx - R, cx + R], [cy, cy], color=WHITE, lw=1.4, alpha=0.7)
    for r in (0.5, 1.0, 2.0):                                  # constant resistance
        ax.add_patch(Circle((cx + R * r / (1 + r), cy), R / (1 + r), fill=False, ec=WHITE, lw=1.4, alpha=0.7))
    t = np.linspace(0, 2 * np.pi, 400)
    for x in (0.5, 1.0, 2.0):                                  # constant reactance arcs
        for s in (1, -1):
            gx, gy = 1 + np.cos(t) / x, s / x + np.sin(t) / x
            inside = gx ** 2 + gy ** 2 <= 1.0
            ax.plot(np.where(inside, cx + R * gx, np.nan), np.where(inside, cy + R * gy, np.nan),
                    color=WHITE, lw=1.4, alpha=0.7)
    th = np.linspace(0.3, 2.6, 80)                             # a trace spiralling inwards
    rho = 0.85 - 0.25 * (th - 0.3) / 2.3
    ax.plot(cx + R * rho * np.cos(th), cy + R * rho * np.sin(th), color=WHITE, lw=3.0)
    return finish(fig, ax, "SnP", path)


def draw_range(path: Path) -> Path:
    """Antenna range: a dish facing a probe across the range."""
    fig, ax = tile(COLOURS["range"])
    y = np.linspace(-0.17, 0.17, 100)
    ax.plot(0.24 + 2.2 * y ** 2, 0.58 + y, color=WHITE, lw=3.4)        # reflector
    ax.plot([0.30, 0.20], [0.58, 0.58], color=WHITE, lw=2.0)           # feed arm
    ax.add_patch(Rectangle((0.74, 0.52), 0.06, 0.12, color=WHITE))      # probe
    for k in range(3):                                                  # waves
        ax.plot([0.40 + 0.09 * k] * 2, [0.50, 0.66], color=WHITE, lw=1.8, alpha=0.7)
    return finish(fig, ax, "Range", path)


def make_app_icon(source: Path, path: Path) -> Path:
    """Square crop of GBT_2.ico (the telescope), resized to 256 px."""
    from PIL import Image
    image = Image.open(str(source)).convert("RGBA")
    w, h = image.size
    side = min(w, h)
    left = max(0, int(0.12 * w))                 # the dish sits left of centre
    left = min(left, w - side)
    square = image.crop((left, 0, left + side, side))
    square.resize((SIZE_PX, SIZE_PX), Image.LANCZOS).save(str(path))
    return path


ICONS = {
    "atr_icon.png": draw_atr,
    "fits_icon.png": draw_fits,
    "franks_icon.png": draw_franks,
    "csvicon.png": draw_csv,
    "rands_icon.png": draw_sft,
    "snp_icon.png": draw_snp,
    "orbitfr_icon.png": draw_range,
}


# %% Entry point
def main(argv=None) -> list:
    """Write every icon; returns the list of files written."""
    argv = sys.argv[1:] if argv is None else argv
    here = Path(__file__).resolve().parent.parent / "assets"
    out = Path(argv[0]) if argv else here
    out.mkdir(parents=True, exist_ok=True)
    written = [make_app_icon(here / "GBT_2.ico", out / "app_icon.png")]
    for name, draw in ICONS.items():
        written.append(draw(out / name))
    for path in written:
        print("wrote", path)
    return written


if __name__ == "__main__":
    main()
