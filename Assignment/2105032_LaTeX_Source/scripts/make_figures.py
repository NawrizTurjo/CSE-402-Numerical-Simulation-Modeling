"""Generate all figures for the Gradient Methods Beamer deck (vector PDFs).

One visual system for every plot:
  * contour maps: soft single-hue filled levels + fine level lines,
  * iterates: orange path with white-edged markers, start = dark square,
    minimiser = star,
  * series colours (fixed order, colour-blind safe): blue, orange, violet,
    then aqua (always legend-labelled),
  * recessive axes: no top/right spines, faint grid, ink-coloured text.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LightSource, LinearSegmentedColormap

OUT = Path(__file__).resolve().parent.parent / "figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
PREVIEW = os.environ.get("PREVIEW_DIR")
if PREVIEW:
    Path(PREVIEW).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
INK, INK2, MUTED, GRIDC = "#1F2937", "#4B5563", "#9CA3AF", "#E7E5E0"
PAPER = "#FBFAF7"   # slide background, so figures sit flush on the slide
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#4a3aa7", "#1baf7a"   # categorical order
BAD = "#e34948"                                              # instability / failure
STAR = "#0f766e"                                             # minimiser landmark
PATH = C2

SEQ = LinearSegmentedColormap.from_list("gd_seq", ["#8fb8ea", "#c9ddf6", "#eaf2fc", "#fbfdff"])
SURF = LinearSegmentedColormap.from_list("gd_surf", ["#1e3a8a", "#2a78d6", "#8fb8ea", "#f1f5f9", "#fdba74"])

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


def save(fig, name, tight=True):
    kw = dict(transparent=False, facecolor=PAPER, dpi=300)
    if tight:
        kw["bbox_inches"] = "tight"
        kw["pad_inches"] = 0.04
    fig.savefig(OUT / name, **kw)
    if PREVIEW:
        fig.savefig(Path(PREVIEW) / (name[:-4] + ".png"), dpi=90, **kw)
    plt.close(fig)
    print("saved", name)


def plot_grid(ax):
    ax.grid(True, which="major")
    ax.set_axisbelow(True)


def contour_map(ax, X, Y, Z, levels, lines=True):
    """Soft filled contour map with fine level lines (colour by level index)."""
    levels = np.asarray(levels, float)
    zmin = float(np.nanmin(Z))
    if levels[0] > zmin:          # fill the basin below the first level too
        levels = np.concatenate(([zmin - 1e-9 * (1 + abs(zmin))], levels))
    norm = BoundaryNorm(levels, SEQ.N, extend="max")
    ax.contourf(X, Y, Z, levels=levels, cmap=SEQ, norm=norm, extend="max", zorder=0)
    ax.set_rasterization_zorder(0.5)   # rasterise only the fills; lines/text stay vector
    if lines:
        ax.contour(X, Y, Z, levels=levels[1:], colors="#5b7fae", linewidths=0.35, alpha=0.55)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(MUTED)


def iterates(ax, P, color=PATH, ms=3.6, lw=1.6, label=None, zorder=4, arrows=False):
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
    """Small white-boxed text label."""
    ax.text(x, y, text, color=color, fontsize=kw.pop("fontsize", 8), zorder=9,
            bbox=dict(boxstyle="round,pad=0.25", fc=PAPER, ec="none", alpha=0.9), **kw)


# ---------------------------------------------------------------- shared problems
A2 = np.array([[3.0, 1.0], [1.0, 3.0]])
B2 = np.array([4.0, 4.0])
X0Q = np.array([-1.5, 2.0])
FSTAR_Q = -4.0


def fq(p):
    p = np.asarray(p)
    return 0.5 * np.einsum("i...,ij,j...->...", p, A2, p) - np.einsum("i,i...->...", B2, p)


def gq(p):
    return A2 @ p - B2


def gd(grad, x0, alpha, n):
    xs = [np.asarray(x0, float)]
    for _ in range(n):
        xs.append(xs[-1] - alpha * grad(xs[-1]))
    return np.array(xs)


def grid(xl, yl, n=300):
    return np.meshgrid(np.linspace(*xl, n), np.linspace(*yl, n))


def quad_levels(Z):
    return FSTAR_Q + np.geomspace(0.05, max(Z.max() - FSTAR_Q, 1), 22)


def log_levels(Z, n=25, lo=None):
    zmin = Z.min()
    lo = lo if lo is not None else max((Z.max() - zmin) * 1e-3, 1e-3)
    return zmin + np.geomspace(lo, Z.max() - zmin, n)


def f_quartic(x, y):
    return x**4 + y**4 - 4 * x * y


def g_quartic(p):
    return np.array([4 * p[0] ** 3 - 4 * p[1], 4 * p[1] ** 3 - 4 * p[0]])


# ---------------------------------------------------------------- 1
def fig_landscape_contour():
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    X, Y = grid((-2, 2), (-2, 2))
    Z = f_quartic(X, Y)
    contour_map(ax, X, Y, Z, log_levels(Z, 26, 0.05))
    gx, gy = np.meshgrid(np.linspace(-1.85, 1.85, 13), np.linspace(-1.85, 1.85, 13))
    G = g_quartic(np.array([gx, gy]))
    nrm = np.hypot(G[0], G[1]) + 1e-12
    ax.quiver(gx, gy, -G[0] / nrm, -G[1] / nrm, color=INK2, scale=26, width=0.0045,
              headwidth=4, headlength=4.5, alpha=0.85, zorder=3)
    ax.plot(0, 0, "X", color=BAD, ms=11, mec="white", mew=1.0, zorder=6)
    mark_min(ax, (1, 1))
    mark_min(ax, (-1, -1))
    tag(ax, 0.12, -0.28, "saddle, f = 0", color=BAD)
    tag(ax, 1.08, 1.2, "min, f = −2", color=STAR)
    tag(ax, -1.9, -1.32, "min, f = −2", color=STAR)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$-\nabla f$ on $f=x^4+y^4-4xy$")
    save(fig, "fig_landscape_contour.pdf")


# ---------------------------------------------------------------- 2
def fig_landscape_surface():
    fig = plt.figure(figsize=(5.0, 3.8))
    ax = fig.add_subplot(projection="3d")
    U, V = np.meshgrid(np.linspace(-2.1, 2.1, 160), np.linspace(-0.7, 0.7, 80))
    X, Y = (U + V) / np.sqrt(2), (U - V) / np.sqrt(2)
    Z = f_quartic(X, Y)
    ls = LightSource(azdeg=315, altdeg=40)
    rgb = ls.shade(Z, cmap=SURF, vert_exag=0.6, blend_mode="soft", vmin=-2, vmax=7)
    ax.plot_surface(X, Y, Z, facecolors=rgb, linewidth=0, antialiased=True, rstride=1, cstride=1,
                    shade=False, rasterized=True)
    ax.scatter([0], [0], [0.25], marker="X", color=BAD, s=90, edgecolor="white", linewidths=0.8,
               depthshade=False, zorder=10)
    ax.scatter([1, -1], [1, -1], [-1.75, -1.75], marker="*", color=STAR, s=180, edgecolor="white",
               linewidths=0.8, depthshade=False, zorder=10)
    ax.set_zlim(-2.6, 6.5)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_zticks([-2, 0, 2, 4, 6])
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))  # transparent panes
        a.pane.set_edgecolor(GRIDC)
        a._axinfo["grid"]["color"] = GRIDC
        a._axinfo["grid"]["linewidth"] = 0.5
    ax.set_xlabel("x", labelpad=-4)
    ax.set_ylabel("y", labelpad=-4)
    ax.set_zlabel("f", labelpad=-6)
    ax.tick_params(labelsize=7, pad=-2)
    ax.view_init(elev=26, azim=-58)
    save(fig, "fig_landscape_surface.pdf")


# ---------------------------------------------------------------- 3
def fig_gd_frames():
    xs = gd(gq, X0Q, 0.2, 7)
    xl, yl = (-2, 2.5), (-0.5, 3)
    X, Y = grid(xl, yl)
    Z = fq(np.array([X, Y]))
    levels = FSTAR_Q + np.geomspace(0.05, 14, 22)
    for k in range(8):
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        fig.subplots_adjust(left=0.10, right=0.97, bottom=0.13, top=0.92)
        contour_map(ax, X, Y, Z, levels)
        if k:
            iterates(ax, xs[: k + 1], color=INK2, ms=3.4, lw=1.3, arrows=True)
        mark_start(ax, xs[0])
        mark_min(ax, (1, 1))
        ax.plot(*xs[k], "o", color=PATH, ms=10, mec="white", mew=1.4, zorder=8)
        g = gq(xs[k])
        d = -g / (np.linalg.norm(g) + 1e-12) * min(0.75, 0.3 * np.linalg.norm(g))
        ax.annotate("", xy=xs[k] + d, xytext=xs[k], zorder=9,
                    arrowprops=dict(arrowstyle="-|>", color=PATH, lw=2.4, mutation_scale=14))
        tag(ax, 0.03, 0.06, f"k = {k}    f = {fq(xs[k]):.3f}", fontsize=9, transform=ax.transAxes)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(r"orange arrow: $-\nabla f(x_k)$     star: $x^\star=(1,1)$", fontsize=9,
                     fontweight="normal", color=INK2)
        save(fig, f"fig_gd_frame_{k}.pdf", tight=False)


# ---------------------------------------------------------------- 4
def fig_stepsize_compare():
    cases = [(0.05, r"$\alpha=0.05$: too small", C1),
             (1 / 3, r"$\alpha=1/3$: optimal", STAR),
             (0.48, r"$\alpha=0.48$: near critical", C2),
             (0.52, r"$\alpha=0.52$: divergent", BAD)]
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.5), sharex=True, sharey=True)
    xl, yl = (-2, 2.5), (-0.5, 3)
    X, Y = grid(xl, yl, 220)
    Z = fq(np.array([X, Y]))
    for ax, (a, t, c) in zip(axs.flat, cases):
        contour_map(ax, X, Y, Z, quad_levels(Z))
        xs = gd(gq, X0Q, a, 15)
        iterates(ax, xs, color=c, ms=3, lw=1.3)
        mark_start(ax, xs[0])
        mark_min(ax, (1, 1), ms=12)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        ax.set_title(t, fontsize=9, color=c)
    for ax in axs[1]:
        ax.set_xlabel("x")
    for ax in axs[:, 0]:
        ax.set_ylabel("y")
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, "fig_stepsize_compare.pdf")


# ---------------------------------------------------------------- 5
def fig_amplification():
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    a = np.linspace(0, 0.6, 600)
    r2, r4 = np.abs(1 - 2 * a), np.abs(1 - 4 * a)
    ax.axvspan(0.5, 0.6, color=BAD, alpha=0.09, lw=0)
    ax.plot(a, r2, color=C1, lw=1.4, ls=(0, (4, 2)), label=r"$|1-2\alpha|$  ($\lambda=2$)")
    ax.plot(a, r4, color=C3, lw=1.4, ls=(0, (4, 2)), label=r"$|1-4\alpha|$  ($\lambda=4$)")
    ax.plot(a, np.maximum(r2, r4), color=INK, lw=2.6, label=r"$\rho(\alpha)$ = max of the two")
    ax.axhline(1, color=BAD, lw=1.0)
    ax.axvline(1 / 3, color=MUTED, ls=":", lw=1)
    ax.axvline(0.5, color=MUTED, ls=":", lw=1)
    ax.plot(1 / 3, 1 / 3, "o", color=PATH, ms=9, mec="white", mew=1.4, zorder=5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=7.5, handlelength=1.8,
              columnspacing=1.0, framealpha=1)
    ax.text(0.10, 1.03, r"stability boundary $\rho=1$", color=BAD, ha="left", fontsize=8)
    ax.text(0.55, 1.18, "unstable", color=BAD, ha="center", fontsize=8.5, fontweight="bold")
    ax.annotate(r"$\alpha^\star=\frac{2}{\mu+L}=1/3$,  $\rho^\star=1/3$", xy=(1 / 3, 1 / 3),
                xytext=(0.012, 0.06), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-|>", color=PATH, lw=1.2, shrinkB=6))
    ax.text(0.5, -0.1, r"$2/L$", ha="center", va="top", color=INK2, fontsize=9,
            transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1.5)
    ax.set_xlabel(r"step size $\alpha$")
    ax.set_ylabel(r"contraction factor $|1-\alpha\lambda|$")
    plot_grid(ax)
    save(fig, "fig_amplification.pdf")


# ---------------------------------------------------------------- 6/7
def exact_ls(gamma, n):
    Ad = np.array([1.0, gamma])
    x = np.array([gamma, 1.0])
    xs = [x]
    for _ in range(n):
        g = Ad * x
        if g @ g > 0:
            x = x - (g @ g) / (g @ (Ad * g)) * g
        xs.append(x)
    return np.array(xs)


def fig_zigzag():
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for ax, gamma in zip(axs, (10, 100)):
        xs = exact_ls(gamma, 20)
        xl = (-0.06 * gamma, 1.06 * gamma)
        yl = (-1.3, 1.3)
        X, Y = grid(xl, yl, 400)
        Z = 0.5 * (X**2 + gamma * Y**2)
        contour_map(ax, X, Y, Z, np.geomspace(0.05, Z.max(), 22))
        iterates(ax, xs, ms=3, lw=1.3)
        mark_start(ax, xs[0])
        mark_min(ax, (0, 0), ms=12)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        ax.set_title(rf"$\gamma={gamma}$", fontsize=10)
        tag(ax, 0.97, 0.05, "axes not to scale", color=INK2, fontsize=7, ha="right", transform=ax.transAxes)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_zigzag.pdf")


def fig_error_curves():
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ks = np.arange(0, 401)
    for gamma, c in ((10, C1), (100, C3)):
        xs = exact_ls(gamma, 400)
        f = 0.5 * (xs[:, 0] ** 2 + gamma * xs[:, 1] ** 2)
        rel = np.maximum(f / f[0], 1e-300)
        r = (gamma - 1) / (gamma + 1)
        ax.semilogy(ks, r ** (2 * ks), color=c, lw=4.5, alpha=0.22, solid_capstyle="butt")
        ax.semilogy(ks[::8], rel[::8], "o", color=c, ms=3.2, mec="white", mew=0.5)
        print(f"   gamma={gamma}: {int(np.argmax(rel <= 1e-6))} iterations to 1e-6 (exact line search)")
    ax.axhline(1e-6, color=INK2, lw=0.9, ls=(0, (3, 2)))
    ax.text(395, 2e-6, r"$10^{-6}$", ha="right", va="bottom", color=INK2, fontsize=8)
    ax.text(40, 1e-11, r"$\gamma=10$", color=C1, fontsize=9, fontweight="bold")
    ax.text(250, 3e-4, r"$\gamma=100$", color=C3, fontsize=9, fontweight="bold")
    ax.text(0.98, 0.97, "dots: measured\nband: $((\\gamma-1)/(\\gamma+1))^{2k}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color=INK2)
    ax.set_ylim(1e-14, 3)
    ax.set_xlim(0, 400)
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$(f_k-f^\star)/(f_0-f^\star)$")
    plot_grid(ax)
    save(fig, "fig_error_curves.pdf")


# ---------------------------------------------------------------- 8
def fig_order():
    gamma = 10
    Ad = np.array([1.0, gamma])
    x = np.array([10.0, 1.0])
    al = 2 / (1 + gamma)
    e = [np.linalg.norm(x)]
    for _ in range(60):
        x = x - al * Ad * x
        e.append(np.linalg.norm(x))
    le = np.log10(np.array(e))
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    x0, y0 = le[30], le[31]
    xr = np.array([le.min() - 0.3, le.max() + 0.3])
    ax.plot(xr, y0 + 1 * (xr - x0), color=C1, lw=1.6)
    ax.plot(xr, y0 + 2 * (xr - x0), color=MUTED, ls=(0, (4, 2)), lw=1.4)
    ax.plot(le[:-1], le[1:], "o", color=PATH, ms=4.5, mec="white", mew=0.6, zorder=4)
    ax.text(0.04, 0.96, "slope 1: linear order (GD data)", color=C1, fontsize=8.5, fontweight="bold",
            transform=ax.transAxes, va="top")
    ax.text(0.04, 0.89, "slope 2: quadratic order (reference)", color=INK2, fontsize=8.5,
            transform=ax.transAxes, va="top")
    ax.set_xlim(xr)
    ax.set_ylim(le.min() - 0.5, le.max() + 0.5)
    ax.set_xlabel(r"$\log_{10} e_k$")
    ax.set_ylabel(r"$\log_{10} e_{k+1}$")
    ax.set_title(r"GD on $\frac{1}{2}(x^2+10y^2)$, $\alpha=2/11$", fontsize=9)
    plot_grid(ax)
    save(fig, "fig_order.pdf")


# ---------------------------------------------------------------- 9
def fig_scaling():
    xd = np.array([1.0, 2.0, 3.0])
    yd = np.array([2.0, 3.0, 5.0])
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.1))
    specs = (
        (axs[0], xd, r"raw feature $x$:  $\kappa\approx46$", (1.5, 1 / 3), (-0.3, 2.5), (-0.5, 3.0)),
        (axs[1], xd - 2, r"centred $x-\bar x$:  $\kappa=1.5$", (1.5, 10 / 3), (-0.3, 2.5), (-0.5, 4.5)),
    )
    for ax, xv, title, opt, wl, bl in specs:
        def loss(w, b):
            r = w[..., None] * xv + b[..., None] - yd
            return (r**2).sum(-1) / (2 * len(xv))

        def grad(p):
            r = p[0] * xv + p[1] - yd
            return np.array([(r * xv).mean(), r.mean()])

        W, Bb = grid(wl, bl, 300)
        Z = loss(W, Bb)
        contour_map(ax, W, Bb, Z, Z.min() + np.geomspace(0.005, Z.max() - Z.min(), 24))
        path = gd(grad, [0, 0], 0.3, 40)
        iterates(ax, path, ms=2.8, lw=1.3)
        mark_start(ax, (0, 0))
        mark_min(ax, opt, ms=13)
        ax.set_xlim(wl)
        ax.set_ylim(bl)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("w")
        ax.set_ylabel("b")
        print(f"   scaling final point after 40 its: {path[-1].round(4)} (optimum {np.round(opt, 4)})")
    tag(axs[0], 0.97, 0.05, "40 steps, α = 0.3", color=INK2, fontsize=7.5, ha="right",
        transform=axs[0].transAxes)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_scaling.pdf")


# ---------------------------------------------------------------- 10
def fig_momentum():
    mu, L = 1.0, 100.0
    Ad = np.array([mu, L])

    def grad(p):
        return Ad * p

    x0 = np.array([10.0, 1.0])
    N = 300
    gdp = gd(grad, x0, 2 / (mu + L), N)
    a_hb = 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    b_hb = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
    hb = [x0, x0 - a_hb * grad(x0)]
    for _ in range(N - 1):
        hb.append(hb[-1] - a_hb * grad(hb[-1]) + b_hb * (hb[-1] - hb[-2]))
    hb = np.array(hb)
    bn = (np.sqrt(L / mu) - 1) / (np.sqrt(L / mu) + 1)
    xn = [x0]
    y = x0.copy()
    for _ in range(N):
        xnew = y - grad(y) / L
        y = xnew + bn * (xnew - xn[-1])
        xn.append(xnew)
    xn = np.array(xn)
    methods = ((gdp, C1, "GD"), (hb, C2, "heavy ball"), (xn, C3, "Nesterov"))

    fig = plt.figure(figsize=(6.8, 3.3))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1], hspace=0.12, wspace=0.28)
    xl, yl = (-1, 11), (-2.6, 2.6)
    X, Y = grid(xl, yl, 300)
    Z = 0.5 * (mu * X**2 + L * Y**2)
    for i, (P, c, lab) in enumerate(methods):
        ax = fig.add_subplot(gs[i, 0])
        contour_map(ax, X, Y, Z, np.geomspace(0.05, Z.max(), 20))
        iterates(ax, P[:41], color=c, ms=2.0, lw=0.9)
        mark_min(ax, (0, 0), ms=10)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(labelsize=7)
        tag(ax, 0.985, 0.08, lab, color=c, fontsize=8, ha="right", transform=ax.transAxes,
            fontweight="bold")
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("x   (first 40 steps; axes not to scale)", fontsize=8)
        if i == 1:
            ax.set_ylabel("y")
    ax = fig.add_subplot(gs[:, 1])
    for P, c, lab in methods:
        nr = np.maximum(np.linalg.norm(P, axis=1), 1e-300)
        ax.semilogy(nr, color=c, lw=1.8, label=lab)
        hit = nr < 1e-6 * nr[0]
        print(f"   momentum {lab}: ||x_k|| < 1e-6 ||x_0|| at k = {int(np.argmax(hit)) if hit.any() else '>300'}")
    ax.text(300, 0.25, "GD", color=C1, ha="right", fontsize=8.5, fontweight="bold")
    ax.text(118, 1e-11, "heavy ball", color=C2, fontsize=8.5, fontweight="bold")
    ax.text(205, 1e-7, "Nesterov", color=C3, fontsize=8.5, fontweight="bold")
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$\|x_k\|$")
    ax.set_xlim(0, N)
    ax.set_ylim(1e-12, 30)
    plot_grid(ax)
    save(fig, "fig_momentum.pdf")


# ---------------------------------------------------------------- 11
def rosen(p):
    return (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2


def rosen_g(p):
    return np.array([-2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] ** 2), 200 * (p[1] - p[0] ** 2)])


def fig_rosenbrock():
    x0 = np.array([-1.2, 1.0])
    x = x0.copy()
    gp = [x.copy()]
    for _ in range(2000):
        g = rosen_g(x)
        t = 1.0
        f0, gg = rosen(x), g @ g
        while rosen(x - t * g) > f0 - 1e-4 * t * gg and t > 1e-16:
            t *= 0.5
        x = x - t * g
        gp.append(x.copy())
    gp = np.array(gp)
    x = x0.copy()
    xprev = x0.copy()
    mp = [x.copy()]
    with np.errstate(all="ignore"):
        for _ in range(2000):
            y = x + 0.9 * (x - xprev)
            xprev = x
            x = y - 1e-3 * rosen_g(y)
            mp.append(x.copy())
    mp = np.array(mp)
    fg = np.array([rosen(p) for p in gp])
    fm = np.array([rosen(p) for p in mp])
    print(f"   rosenbrock final f: GD-Armijo {fg[-1]:.3e}, momentum {fm[-1]:.3e}")
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0))
    ax = axs[0]
    xl, yl = (-1.5, 1.5), (-0.5, 1.5)
    X, Y = grid(xl, yl, 400)
    Z = rosen(np.array([X, Y]))
    contour_map(ax, X, Y, Z, np.geomspace(1e-2, Z.max(), 26))
    ax.plot(*gp.T, "-", color=C1, lw=1.5, label="GD + Armijo", zorder=4)
    ax.plot(*mp.T, "-", color=C2, lw=1.5, label="momentum", zorder=5)
    mark_start(ax, x0)
    mark_min(ax, (1, 1), ms=13)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Rosenbrock valley", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=7.5)
    ax = axs[1]
    ax.semilogy(np.maximum(fg, 1e-30), color=C1, lw=1.8)
    ax.semilogy(np.maximum(fm, 1e-30), color=C2, lw=1.8)
    ax.text(1300, fg[1300] * 3, "GD + Armijo", color=C1, ha="center", va="bottom", fontsize=8.5,
            fontweight="bold")
    ax.text(1500, fm[1500] * 0.04, "momentum", color=C2, ha="center", va="top", fontsize=8.5,
            fontweight="bold")
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$f_k$")
    ax.set_xlim(0, 2000)
    plot_grid(ax)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_rosenbrock.pdf")


# ---------------------------------------------------------------- 12
def fig_linesearch():
    lam = np.linspace(1, 100, 50)
    L = 100.0
    x0 = np.ones(50)
    N = 300

    def f(x):
        return 0.5 * (lam * x**2).sum()

    def run(step):
        x = x0.copy()
        fs = [f(x)]
        xp = gp = None
        for k in range(N):
            g = lam * x
            a = step(x, g, xp, gp)
            xp, gp = x, g
            x = x - a * g
            fs.append(f(x))
        return np.array(fs)

    def armijo(x, g, xp, gp):
        t, f0, gg = 1.0, f(x), g @ g
        while f(x - t * g) > f0 - 1e-4 * t * gg and t > 1e-16:
            t *= 0.5
        return t

    def bb(x, g, xp, gp):
        if xp is None:
            return 1 / L
        s, y = x - xp, g - gp
        d = s @ y
        return (s @ s) / d if d > 0 else 1 / L

    res = {"fixed 1/L": run(lambda *a: 1 / L), "Armijo": run(armijo), "Barzilai–Borwein": run(bb)}
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for (lab, fs), c in zip(res.items(), (C1, C2, C3)):
        ax.semilogy(np.maximum(fs, 1e-30), color=c, lw=1.8 if c != C3 else 1.2)
        print(f"   linesearch {lab}: final f={fs[-1]:.2e}")
    ax.text(298, 4e-3, "fixed 1/L", color=C1, ha="right", va="bottom", fontsize=8.5, fontweight="bold")
    ax.text(298, 2e-7, "Armijo", color=C2, ha="right", va="top", fontsize=8.5, fontweight="bold")
    ax.text(150, 1e-16, "Barzilai–Borwein\n(nonmonotone)", color=C3, ha="left", fontsize=8.5,
            fontweight="bold")
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$f_k-f^\star$")
    ax.set_xlim(0, N)
    ax.set_ylim(1e-20, 1e4)
    plot_grid(ax)
    save(fig, "fig_linesearch.pdf")


# ---------------------------------------------------------------- 13
def fig_sgd():
    rng = np.random.default_rng(0)
    theta, sigma, K, R = 20.0, 0.5, 500, 500
    xi = rng.normal(theta, sigma, size=(R, K))

    def run(sched):
        x = np.zeros(R)
        out = np.empty((R, K + 1))
        out[:, 0] = 0
        for k in range(K):
            a = sched(k)
            x = (1 - a) * x + a * xi[:, k]
            out[:, k + 1] = x
        return out

    c = run(lambda k: 0.1)
    h = run(lambda k: 1.0 / (k + 1))
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
    ax = axs[0]
    ks = np.arange(K + 1)
    ax.axhspan(theta - 2 * np.sqrt(0.1 * sigma**2 / 1.9), theta + 2 * np.sqrt(0.1 * sigma**2 / 1.9),
               color=C2, alpha=0.10, lw=0)
    ax.plot(ks, c[0], color=C2, lw=0.9)
    ax.plot(ks, h[0], color=C1, lw=1.6)
    ax.axhline(theta, color=INK, ls=(0, (3, 2)), lw=0.9)
    ax.text(495, 21.15, r"$\alpha=0.1$ (noise ball $\pm2\,$sd)", color=C2, ha="right", fontsize=8,
            fontweight="bold")
    ax.text(495, 19.0, r"$\alpha_k=1/(k+1)$", color=C1, ha="right", fontsize=8, fontweight="bold")
    ax.set_ylim(18.5, 21.5)
    ax.set_xlim(0, K)
    ax.set_xlabel("k")
    ax.set_ylabel(r"$x_k$  (one run)")
    plot_grid(ax)
    ax = axs[1]
    kk = np.arange(1, K + 1)
    mc = ((c - theta) ** 2).mean(0)
    mh = ((h - theta) ** 2).mean(0)
    ax.loglog(kk, mc[1:], color=C2, lw=1.8)
    ax.loglog(kk, mh[1:], color=C1, lw=1.8)
    ax.axhline(0.1 * sigma**2 / 1.9, color=C2, ls=(0, (3, 2)), lw=1)
    ax.loglog(kk, sigma**2 / kk, color=C1, ls=(0, (3, 2)), lw=1)
    ax.text(1.2, 0.1 * sigma**2 / 1.9 * 1.4, r"$\alpha\sigma^2/(2-\alpha)$", color=C2, fontsize=8)
    ax.text(30, 2e-3, r"$\sigma^2/k$", color=C1, fontsize=8.5)
    ax.set_xlabel("k")
    ax.set_ylabel(r"$\mathbb{E}[(x_k-\theta)^2]$  (500 runs)")
    plot_grid(ax)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_sgd.pdf")


# ---------------------------------------------------------------- 14
def fig_cooling_loss():
    t = np.array([1.0, 2.0])
    y = np.array([0.6, 0.36])

    def E(k):
        return np.exp(-np.multiply.outer(k, t))

    def J(k):
        return 0.5 * ((E(k) - y) ** 2).sum(-1)

    def dJ(k):
        return ((E(k) - y) * (-t * E(k))).sum(-1)

    def d2J(k):
        return (t**2 * E(k) ** 2 + (E(k) - y) * t**2 * E(k)).sum(-1)

    ks = np.linspace(0, 10, 1000)
    kstar = np.log(5 / 3)
    plateau = 0.5 * (y**2).sum()

    def it(k0, n):
        k = [k0]
        for _ in range(n):
            k.append(k[-1] - 0.5 * float(dJ(np.array(k[-1]))))
        return np.array(k)

    a, b = it(0.0, 8), it(6.0, 50)
    print(f"   cooling: plateau={plateau:.4f}, k*={kstar:.4f}, GD from 0 -> {a[-1]:.4f}, from 6 -> {b[-1]:.4f}")
    d = d2J(ks)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
    axs[1].axvspan(ks[d < 0].min(), 10, color=BAD, alpha=0.07, lw=0)
    ax = axs[0]
    ax.plot(ks, J(ks), color=INK, lw=2.0)
    ax.axhline(plateau, color=INK2, ls=(0, (3, 2)), lw=0.9)
    ax.plot(a, J(a), "-", color=C1, lw=1.2, zorder=4)
    ax.plot(a, J(a), "o", color=C1, ms=4.5, mec="white", mew=0.7, zorder=5)
    ax.plot(b, J(b), "s", color=C2, ms=4.5, mec="white", mew=0.6, zorder=5)
    ax.plot(kstar, 0, "*", color=STAR, ms=14, mec="white", mew=1, clip_on=False, zorder=6)
    ax.text(1.05, 0.004, r"$k^\star=\ln(5/3)$", color=STAR, fontsize=8.5)
    ax.text(9.8, plateau + 0.012, "plateau 0.2448", color=INK2, ha="right", fontsize=8)
    ax.annotate(r"GD from $k_0=0$", xy=(a[1], float(J(np.array(a[1])))), xytext=(2.3, 0.075),
                color=C1, fontsize=8.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=C1, lw=1))
    ax.text(6.0, 0.205, r"GD from $k_0=6$: stuck", color=C2, fontsize=8.5, fontweight="bold",
            ha="center")
    ax.set_xlabel("k")
    ax.set_ylabel("J(k)")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.01, 0.33)
    plot_grid(ax)
    ax = axs[1]
    ax.plot(ks, d, color=INK, lw=2.0)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.fill_between(ks, d, 0, where=d < 0, color=BAD, alpha=0.35, lw=0)
    ax.text(5.5, -0.12, "J'' < 0: nonconvex", color=BAD, fontsize=8.5, ha="center", fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.25, 0.4)
    ax.set_xlabel("k")
    ax.set_ylabel("J''(k)")
    plot_grid(ax)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_cooling_loss.pdf")


# ---------------------------------------------------------------- 15
def fig_fd_error():
    eps = np.finfo(float).eps
    h = np.logspace(-16, 0, 200)
    fwd = np.abs((np.exp(1 + h) - np.exp(1)) / h - np.e)
    cen = np.abs((np.exp(1 + h) - np.exp(1 - h)) / (2 * h) - np.e)
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.loglog(h, eps / h, color=MUTED, ls=(0, (3, 2)), lw=1)
    ax.loglog(h, h, color=C1, ls=(0, (3, 2)), lw=0.9, alpha=0.7)
    ax.loglog(h, h**2, color=C2, ls=(0, (3, 2)), lw=0.9, alpha=0.7)
    ax.loglog(h, np.maximum(fwd, 1e-18), color=C1, lw=1.6)
    ax.loglog(h, np.maximum(cen, 1e-18), color=C2, lw=1.6)
    for xv, c in ((np.sqrt(eps), C1), (eps ** (1 / 3), C2)):
        ax.axvline(xv, color=c, ls=":", lw=1)
    ax.text(2e-16, 1e-10, "round-off\ndominates\n" + r"$\propto\epsilon/h$", color=INK2, fontsize=8)
    ax.text(0.5, 1e-10, "truncation\ndominates\n" + r"$\propto h,\ h^2$", color=INK2, fontsize=8, ha="right")
    ax.text(np.sqrt(eps) * 0.75, 3e-12, r"$\sqrt{\epsilon}$", color=C1, ha="right", fontsize=9)
    ax.text(eps ** (1 / 3) * 1.3, 3e-12, r"$\epsilon^{1/3}$", color=C2, ha="left", fontsize=9)
    ax.text(0.5, 2e-1, "forward", color=C1, ha="right", fontsize=8.5, fontweight="bold")
    ax.text(0.5, 3e-4, "central", color=C2, ha="right", fontsize=8.5, fontweight="bold")
    ax.set_ylim(1e-12, 1e1)
    ax.set_xlim(1e-16, 1)
    ax.set_xlabel("step size h")
    ax.set_ylabel("absolute error in f'(1)")
    plot_grid(ax)
    save(fig, "fig_fd_error.pdf")


# ---------------------------------------------------------------- 16
def fig_hb_spectrum():
    mu, L = 1.0, 100.0
    lam = np.linspace(1, 100, 500)
    a_gd = 2 / (mu + L)
    a = 4 / 121
    b = 81 / 121
    s = 1 + b - a * lam
    disc = s**2 - 4 * b
    rho = np.where(disc < 0, np.sqrt(b), (np.abs(s) + np.sqrt(np.maximum(disc, 0))) / 2)
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.fill_between(lam, rho, 99 / 101, color=C2, alpha=0.10, lw=0)
    ax.plot(lam, np.abs(1 - a_gd * lam), color=C1, lw=1.8)
    ax.plot(lam, rho, color=C2, lw=2.2)
    ax.axhline(99 / 101, color=C1, ls=(0, (3, 2)), lw=1)
    ax.text(50.5, 1.01, r"GD worst case $99/101\approx0.980$", color=C1, ha="center", fontsize=8)
    ax.text(50.5, 0.76, r"heavy ball: $|r|=\sqrt{\beta}=9/11$ for every $\lambda$", color=C2,
            ha="center", fontsize=8, fontweight="bold")
    ax.text(14, 0.5, r"GD: $|1-\alpha\lambda|$", color=C1, fontsize=8.5)
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel(r"eigenvalue $\lambda$  ($\mu=1$, $L=100$)")
    ax.set_ylabel("per-step factor")
    plot_grid(ax)
    save(fig, "fig_hb_spectrum.pdf")


# ---------------------------------------------------------------- 17
def fig_profit():
    def P(x, y):
        return 100 * x + 80 * y - 2 * x**2 - 2 * y**2 - 2 * x * y

    def gP(p):
        return np.array([100 - 4 * p[0] - 2 * p[1], 80 - 4 * p[1] - 2 * p[0]])

    path = [np.zeros(2)]
    for _ in range(8):
        path.append(path[-1] + 0.25 * gP(path[-1]))
    path = np.array(path)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    X, Y = grid((-2, 30), (-2, 24), 300)
    Z = P(X, Y)
    levels = np.sort(1400 - np.geomspace(2, 1400 - Z.min(), 24))
    # maximisation: invert so the peak is the deepest colour
    contour_map(ax, X, Y, -Z, np.sort(-levels))
    iterates(ax, path, ms=4.5, lw=1.6)
    offs = [(-4, -14), (6, 2), (-14, -4), (-12, 4)]
    for i in range(4):
        ax.annotate(f"$x_{i}$", path[i], xytext=offs[i], textcoords="offset points",
                    color=INK, fontsize=9, fontweight="bold", zorder=9)
    mark_min(ax, (20, 10), ms=16)
    tag(ax, 21.2, 7.4, "P* = 1400", color=STAR, fontsize=8.5)
    ax.set_aspect("equal")
    ax.set_xlim(-2, 30)
    ax.set_ylim(-2, 24)
    ax.set_xlabel("x (product 1)")
    ax.set_ylabel("y (product 2)")
    ax.set_title(r"gradient ascent, $\alpha=1/4$", fontsize=9.5)
    save(fig, "fig_profit.pdf")


# ---------------------------------------------------------------- 18
def fig_M1_step():
    path = gd(g_quartic, [1.0, 0.0], 0.1, 25)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    xl, yl = (-0.2, 1.6), (-0.5, 1.5)
    X, Y = grid(xl, yl, 300)
    Z = f_quartic(X, Y)
    contour_map(ax, X, Y, Z, -2 + np.geomspace(0.02, Z.max() + 2, 26))
    iterates(ax, path, ms=3.6, lw=1.5)
    mark_start(ax, path[0])
    mark_min(ax, (1, 1))
    ax.annotate(r"$x_1=(0.6,\,0.4)$", path[1], xytext=(-74, -34), textcoords="offset points",
                color=INK, fontsize=9, zorder=9,
                bbox=dict(boxstyle="round,pad=0.25", fc=PAPER, ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
    tag(ax, 1.03, -0.12, r"$x_0=(1,0)$", fontsize=9)
    tag(ax, 1.06, 1.16, "min (1,1)", color=STAR, fontsize=8.5)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"GD with $\alpha=0.1$", fontsize=9.5)
    save(fig, "fig_M1_step.pdf")


# ---------------------------------------------------------------- title art
def fig_title_bg():
    """Transparent contour field + GD path, drawn in light ink for the dark title slide."""
    fig = plt.figure(figsize=(4.6, 4.6))
    ax = fig.add_axes([0, 0, 1, 1])
    X, Y = grid((-2.1, 2.1), (-2.1, 2.1), 400)
    Z = f_quartic(X, Y)
    ax.contour(X, Y, Z, levels=log_levels(Z, 30, 0.03), colors="#bfdbfe", linewidths=0.6, alpha=0.55)
    path = gd(g_quartic, [-0.35, 1.9], 0.04, 40)
    ax.plot(*path.T, "-", color="#fdba74", lw=1.6, alpha=0.95)
    ax.plot(*path.T, "o", color="#fdba74", ms=3, mec="none", alpha=0.95)
    ax.plot(1, 1, "*", color="#5eead4", ms=14, mec="none")
    ax.plot(-1, -1, "*", color="#5eead4", ms=14, mec="none", alpha=0.7)
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    ax.set_axis_off()
    fig.savefig(OUT / "fig_title_bg.pdf", transparent=True)
    plt.close(fig)
    print("saved fig_title_bg.pdf")


def main():
    fig_title_bg()
    for fn in (fig_landscape_contour, fig_landscape_surface, fig_gd_frames, fig_stepsize_compare,
               fig_amplification, fig_zigzag, fig_error_curves, fig_order, fig_scaling,
               fig_momentum, fig_rosenbrock, fig_linesearch, fig_sgd, fig_cooling_loss,
               fig_fd_error, fig_hb_spectrum, fig_profit, fig_M1_step):
        fn()


if __name__ == "__main__":
    main()
