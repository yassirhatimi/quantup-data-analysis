"""Matplotlib styling for QuantLab charts.

Implements a validated, colorblind-safe palette (light mode):
- categorical hues are assigned in FIXED order (never cycled/regenerated),
- sequential = one blue ramp, diverging = blue <-> red with a gray midpoint,
- recessive grid and axes, thin marks, no dual axes.

Call `apply_style()` once per notebook/script before plotting.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots, fixed order (light-surface variant).
PALETTE = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]

# Chart chrome (light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # primary text
SECONDARY = "#52514e"    # secondary text
MUTED = "#898781"        # axis/tick labels
GRID = "#e1e0d9"         # hairline gridlines
BASELINE = "#c3c2b7"     # axis line

# Diverging pair (polarity: negative <-> positive) and its neutral midpoint.
POSITIVE = "#2a78d6"     # blue pole
NEGATIVE = "#e34948"     # red pole
NEUTRAL_MID = "#f0efec"

# One-hue blue ramp for sequential (magnitude) encodings.
SEQUENTIAL_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def sequential_cmap() -> LinearSegmentedColormap:
    """Light-to-dark single-hue colormap for magnitudes (heatmaps etc.)."""
    return LinearSegmentedColormap.from_list("ql_seq", SEQUENTIAL_STEPS)


def diverging_cmap() -> LinearSegmentedColormap:
    """Red -> neutral gray -> blue colormap for signed values (center at 0,
    negative pole = red, positive pole = blue)."""
    return LinearSegmentedColormap.from_list(
        "ql_div", ["#8f1f1f", NEGATIVE, NEUTRAL_MID, POSITIVE, "#104281"]
    )


def mpl_pct(decimals: int = 0):
    """Percent tick formatter for axes holding fractional values (0.05 -> 5%)."""
    from matplotlib.ticker import PercentFormatter

    return PercentFormatter(1.0, decimals=decimals)


def correlation_heatmap(corr, title: str = "Correlation", ax=None):
    """Annotated correlation heatmap on the diverging scale (fixed to [-1, 1])."""
    import matplotlib.pyplot as plt

    n = len(corr)
    if ax is None:
        _, ax = plt.subplots(figsize=(1.1 * n + 2.5, 1.1 * n))
    im = ax.imshow(corr.values, cmap=diverging_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns)
    ax.set_yticks(range(n), corr.index)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                color="#ffffff" if abs(v) > 0.6 else INK,
            )
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    return ax


def apply_style() -> None:
    """Set matplotlib rcParams for the QuantLab look. Call once, then plot."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "figure.figsize": (9, 4.5),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": SECONDARY,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": SECONDARY,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
        }
    )


def end_labels(ax, df, fmt: str = "{name}") -> None:
    """Direct-label each line at its last point (text in ink, not series color)."""
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        ax.annotate(
            " " + fmt.format(name=col, value=series.iloc[-1]),
            xy=(series.index[-1], series.iloc[-1]),
            color=SECONDARY,
            fontsize=9,
            va="center",
        )
