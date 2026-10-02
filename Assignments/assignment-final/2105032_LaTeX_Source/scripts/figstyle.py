"""paper-navy-beamer :: matplotlib style for figures that sit flush on the slides.

Import from your figure script:

    import sys; sys.path.insert(0, "scripts")       # wherever figstyle.py lives
    from figstyle import *                            # palette, rcParams, helpers
    fig, ax = plt.subplots()
    contour_map(ax, X, Y, Z, levels)                   # soft filled contours + fine lines
    iterates(ax, path)                                 # orange path, white-edged markers
    mark_start(ax, path[0]); mark_min(ax, (1, 1))
    tag(ax, x, y, "label")                             # small paper-coloured label
    save(fig, OUT / "fig_name.pdf")

    title_art(f, (-2, 2), (-2, 2), OUT / "fig_title_bg.pdf", path=path)   # title-slide artwork

Visual rules (keep them, they are what makes the deck look consistent):
  * background = slide paper (#FBFAF7) so plots never sit in white boxes;
  * categorical colours in this fixed order: C1 blue, C2 orange, C3 violet, C4 aqua
    (the first three pass colour-blind checks pairwise; aqua is low-contrast, so label it);
  * STAR (teal) marks optima, BAD (red) marks instability / failure;
  * prefer direct labels next to curves over legend boxes; no top/right spines; faint grid;
  * filled contours are rasterised (zorder 0) so PDFs stay small; lines/text stay vector.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402

__all__ = ["plt", "np", "INK", "INK2", "MUTED", "GRIDC", "PAPER", "C1", "C2", "C3", "C4", "BAD", "STAR",
           "PATH", "SEQ", "SURF", "save", "plot_grid", "contour_map", "iterates", "mark_start", "mark_min",
           "tag", "callout", "title_art"]

INK, INK2, MUTED, GRIDC = "#1F2937", "#4B5563", "#9CA3AF", "#E7E5E0"
PAPER = "#FBFAF7"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#4a3aa7", "#1baf7a"
BAD = "#e34948"
STAR = "#0f766e"
PATH = C2

SEQ = LinearSegmentedColormap.from_list("pn_seq", ["#8fb8ea", "#c9ddf6", "#eaf2fc", "#fbfdff"])
SURF = LinearSegmentedColormap.from_list("pn_surf", ["#1e3a8a", "#2a78d6", "#8fb8ea", "#f1f5f9", "#fdba74"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 6,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.figsize": (6.0, 3.2), "figure.facecolor": PAPER,
    "axes.facecolor": PAPER, "savefig.facecolor": PAPER, "legend.facecolor": PAPER,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "text.color": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRIDC, "grid.linewidth": 0.6,
    "legend.frameon": True, "legend.framealpha": 0.92, "legend.edgecolor": "none",
    "legend.fancybox": True, "lines.solid_capstyle": "round",
    "text.usetex": False, "pdf.fonttype": 42,
})


def save(fig, path, tight=True):
    """Save as PDF on the paper background. Use tight=False for animation frames
    that must all have identical size."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(transparent=False, facecolor=PAPER, dpi=300)
    if tight:
        kw.update(bbox_inches="tight", pad_inches=0.08)
    fig.savefig(path, **kw)
    plt.close(fig)
    print("saved", path.name)


def plot_grid(ax):
    ax.grid(True, which="major")
    ax.set_axisbelow(True)


def contour_map(ax, X, Y, Z, levels, lines=True):
    """Soft single-hue filled contours (coloured by level index) with fine level lines.
    The basin below the first level is filled too, so minima never show a white hole."""
    levels = np.asarray(levels, float)
    zmin = float(np.nanmin(Z))
    if levels[0] > zmin:
        levels = np.concatenate(([zmin - 1e-9 * (1 + abs(zmin))], levels))
    norm = BoundaryNorm(levels, SEQ.N, extend="max")
    ax.contourf(X, Y, Z, levels=levels, cmap=SEQ, norm=norm, extend="max", zorder=0)
    ax.set_rasterization_zorder(0.5)
    if lines:
        ax.contour(X, Y, Z, levels=levels[1:], colors="#5b7fae", linewidths=0.35, alpha=0.55)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(MUTED)


def iterates(ax, P, color=PATH, ms=3.6, lw=1.6, label=None, zorder=4, arrows=False):
    P = np.asarray(P)
    ax.plot(*P.T, "-", color=color, lw=lw, zorder=zorder, label=label, alpha=0.95)
    ax.plot(*P.T, "o", color=color, ms=ms, mec="white", mew=0.7, zorder=zorder + 0.1)
    if arrows:
        for i in range(len(P) - 1):
            ax.annotate("", xy=P[i + 1], xytext=P[i], zorder=zorder,
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=3, shrinkB=3,
                                        mutation_scale=9))


def mark_start(ax, p, label=None):
    ax.plot(*p, "s", color=INK, ms=6, mec="white", mew=0.8, zorder=6, label=label)


def mark_min(ax, p, label=None, ms=15):
    ax.plot(*p, "*", color=STAR, ms=ms, mec="white", mew=1.0, zorder=7, label=label)


def tag(ax, x, y, text, color=INK, **kw):
    """Small label on a paper-coloured pad. Pass transform=ax.transAxes for axes coordinates."""
    ax.text(x, y, text, color=color, fontsize=kw.pop("fontsize", 8), zorder=9,
            bbox=dict(boxstyle="round,pad=0.25", fc=PAPER, ec="none", alpha=0.9), **kw)


def callout(ax, point, position, text, color=INK, ha="left"):
    """Label in axes coordinates, with a paper pad and a leader to the data."""
    return ax.annotate(text, xy=point, xytext=position, textcoords="axes fraction",
                       color=color, fontsize=8, ha=ha, va="center", zorder=9,
                       bbox=dict(boxstyle="round,pad=0.25", fc=PAPER, ec="none"),
                       arrowprops=dict(arrowstyle="-", color=color, lw=0.8,
                                       shrinkA=4, shrinkB=6))


def title_art(f, xlim, ylim, out, path=None, n_levels=30, minima=()):
    """Transparent contour field (light ink) for the dark title slide.
    f(X, Y) -> Z; path: optional (k, 2) array drawn in light orange; minima: list of (x, y)."""
    fig = plt.figure(figsize=(4.6, 4.6))
    ax = fig.add_axes([0, 0, 1, 1])
    X, Y = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    Z = f(X, Y)
    zmin = Z.min()
    levels = zmin + np.geomspace(max((Z.max() - zmin) * 1e-3, 1e-3), Z.max() - zmin, n_levels)
    ax.contour(X, Y, Z, levels=levels, colors="#bfdbfe", linewidths=0.6, alpha=0.55)
    if path is not None:
        path = np.asarray(path)
        ax.plot(*path.T, "-", color="#fdba74", lw=1.6, alpha=0.95)
        ax.plot(*path.T, "o", color="#fdba74", ms=3, mec="none", alpha=0.95)
    for m in minima:
        ax.plot(*m, "*", color="#5eead4", ms=14, mec="none")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_axis_off()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, transparent=True)
    plt.close(fig)
    print("saved", out.name)
