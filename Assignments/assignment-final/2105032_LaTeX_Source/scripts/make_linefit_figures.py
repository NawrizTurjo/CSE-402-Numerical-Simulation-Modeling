"""Generate the line-fit figures (weight-height data) for the Gradient Methods deck.

Run:  python scripts/make_figures.py
Set PREVIEW_DIR to also write PNG previews (dpi 80).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import *  # noqa: F401,F403,E402
from figstyle import save as _save  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures" / "generated"
PREV = Path(os.environ["PREVIEW_DIR"]) if os.environ.get("PREVIEW_DIR") else None

X = np.array([0.5, 2.3, 2.9])
Y = np.array([1.4, 1.9, 3.2])
M_FIX = 0.64


def save(fig, name, tight=True):
    if PREV is not None:
        PREV.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREV / (Path(name).stem + ".png"), dpi=80, facecolor=PAPER,
                    **(dict(bbox_inches="tight", pad_inches=0.04) if tight else {}))
    _save(fig, OUT / name, tight=tight)


def gd(grad, x0, alpha, n):
    x = np.asarray(x0, float)
    P = [x.copy()]
    for _ in range(n):
        x = x - alpha * grad(x)
        P.append(x.copy())
    return np.array(P)


def ssr(b, m):
    b, m = np.asarray(b, float)[..., None], np.asarray(m, float)[..., None]
    return ((Y - b - m * X) ** 2).sum(-1)


# ---------------------------------------------------------------- 1 title art
def fig_title():
    """Transparent contour field + GD path, drawn in light ink for the dark title slide."""
    fig = plt.figure(figsize=(4.6, 4.6))
    ax = fig.add_axes([0, 0, 1, 1])
    gx, gy = np.meshgrid(np.linspace(-2.1, 2.1, 400), np.linspace(-2.1, 2.1, 400))
    Z = f_quartic(gx, gy)
    zmin = float(Z.min())
    ax.contour(gx, gy, Z, levels=zmin + np.geomspace(0.03, float(Z.max()) - zmin, 30),
               colors="#bfdbfe", linewidths=0.6, alpha=0.55)
    grad = lambda p: np.array([4 * p[0] ** 3 - 4 * p[1], 4 * p[1] ** 3 - 4 * p[0]])
    path = gd(grad, [-0.35, 1.9], 0.04, 40)
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


# ---------------------------------------------------------------- 2 SSR vs b
def b_grad(b):
    return -2 * (Y - b - M_FIX * X).sum()


B_STAR = (Y - M_FIX * X).mean()
B_ITER = gd(lambda b: np.array(b_grad(b)), 0.0, 0.1, 8)


def fig_ssr_intercept():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bb = np.linspace(-0.5, 2.5, 300)
    ss = lambda b: ssr(b, np.full(np.shape(b), M_FIX))
    plot_grid(ax)
    ax.plot(bb, ss(bb), color=INK2, lw=1.8, zorder=2)
    bk = B_ITER
    pts = np.column_stack([bk, ss(bk)])
    for i in range(len(pts) - 1):
        ax.annotate("", xy=pts[i + 1], xytext=pts[i], zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=C2, lw=0.8, shrinkA=3, shrinkB=3, mutation_scale=7))
    ax.plot(*pts.T, "o", color=C2, ms=4.5, mec="white", mew=0.7, zorder=5)
    mark_start(ax, pts[0])
    mark_min(ax, (B_STAR, float(ss(B_STAR))))
    callout(ax, pts[0], (0.10, 0.83), "b$_0$ = 0")
    callout(ax, (pts[0] + pts[1]) / 2, (0.38, 0.64), "big step", color=C2)
    callout(ax, pts[-2], (0.43, 0.46), "tiny steps\nnear the bottom", color=C2)
    callout(ax, (B_STAR, float(ss(B_STAR))), (0.60, 0.10),
            f"b* = {B_STAR:.2f}", color=STAR)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.9, float(ss(-0.5)) + 0.6)
    ax.set_xlabel("intercept b")
    ax.set_ylabel("SSR")
    save(fig, "fig_ssr_intercept.pdf")


# ---------------------------------------------------------------- 3 line frames
def fig_line_frames():
    ks = [0, 1, 2, 3, 5, 20]
    B = gd(lambda b: np.array(b_grad(b)), 0.0, 0.1, max(ks))
    for i, k in enumerate(ks):
        b = float(B[k])
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        fig.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.95)
        plot_grid(ax)
        w = np.array([0, 3.2])
        ax.plot(w, b + M_FIX * w, color=C2, lw=2.2, zorder=3)
        for xi, yi in zip(X, Y):
            ax.plot([xi, xi], [yi, b + M_FIX * xi], "--", color=BAD, lw=1.1, zorder=3)
        ax.plot(X, Y, "o", color=INK, ms=6.5, mec="white", mew=0.8, zorder=5)
        s = float(ssr(b, M_FIX))
        tag(ax, 0.03, 0.96, f"step {k}: b = {b:.2f}, SSR = {s:.2f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=9)
        ax.set_xlim(0, 3.2)
        ax.set_ylim(0, 3.6)
        ax.set_xlabel("weight")
        ax.set_ylabel("height")
        save(fig, f"fig_line_frame_{i}.pdf", tight=False)


# ---------------------------------------------------------------- 4 lr regimes
def fig_lr_regimes():
    f = lambda x: (x - 3) ** 2 + 2
    cfg = [(0.1, "too small: slow", C1, 12), (0.5, "just right: one step", STAR, 3),
           (0.9, "too big: oscillates", C2, 12), (1.05, "way too big: diverges", BAD, 4)]
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 4.4))
    xx = np.linspace(-1, 7, 300)
    for ax, (a, ttl, col, n) in zip(axs.ravel(), cfg):
        plot_grid(ax)
        ax.plot(xx, f(xx), color=INK2, lw=1.4, zorder=2)
        P = gd(lambda x: 2 * (x - 3), 0.0, a, n)
        ax.plot(P, f(P), "-", color=col, lw=1.2, zorder=4, clip_on=True)
        ax.plot(P, f(P), "o", color=col, ms=4, mec="white", mew=0.6, zorder=5, clip_on=True)
        mark_start(ax, (0, f(0)))
        ax.set_title(f"$\\alpha$ = {a}\n{ttl}", color=col, fontsize=9)
        ax.set_xlim(-1, 7)
        ax.set_ylim(0, 22)
    for ax in axs[1]:
        ax.set_xlabel("x")
    for ax in axs[:, 0]:
        ax.set_ylabel("f(x)")
    fig.tight_layout(pad=1.1, h_pad=1.8, w_pad=1.8)
    save(fig, "fig_lr_regimes.pdf")


# ---------------------------------------------------------------- 5 gradient field
def fig_gradient_field():
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    g = np.linspace(-2, 2, 300)
    XX, YY = np.meshgrid(g, g)
    contour_map(ax, XX, YY, XX ** 2 + YY ** 2, [0.25, 0.5, 1, 1.5, 2.25, 3, 4, 5, 6, 8])
    q = np.linspace(-1.8, 1.8, 11)
    QX, QY = np.meshgrid(q, q)
    U, V = -2 * QX, -2 * QY
    N = np.hypot(U, V)
    N[N == 0] = 1
    ax.quiver(QX, QY, U / N, V / N, color=C1, angles="xy", scale_units="xy", scale=5.5, width=0.005,
              zorder=3)
    p = np.array([1.2, 0.9])
    d = -p / np.hypot(*p)
    ax.annotate("", xy=p + 0.9 * d, xytext=p, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=C2, lw=2.2, mutation_scale=14, shrinkA=0, shrinkB=0))
    ax.plot(*p, "o", color=INK, ms=5, mec="white", zorder=7)
    th = np.linspace(0, 2 * np.pi, 200)
    r = np.hypot(*p)
    ax.plot(r * np.cos(th), r * np.sin(th), color=INK2, lw=1.0, zorder=2)
    ax.set_title("$-\\nabla J \\perp$ contour", color=C2, fontsize=9,
                 loc="center", pad=12)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    save(fig, "fig_gradient_field.pdf")


# ---------------------------------------------------------------- 6 zigzag
def fig_zigzag_scaling():
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0))
    g1, g2 = np.linspace(-2.5, 2.5, 300), np.linspace(-1.5, 1.5, 300)
    XX, YY = np.meshgrid(g1, g2)
    cases = [(1, "round contours: straight in"), (4, "stretched contours: zig-zag")]
    lv = [0.25, 0.5, 1, 2, 3, 4, 6, 8]
    for ax, (c, ttl) in zip(axs, cases):
        contour_map(ax, XX, YY, XX ** 2 + c * YY ** 2, lv)
        P = gd(lambda p: np.array([2 * p[0], 2 * c * p[1]]), (2, 1), 0.2, 14)
        iterates(ax, P, ms=3.2, lw=1.2)
        mark_start(ax, P[0])
        mark_min(ax, (0, 0), ms=12)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        callout(ax, P[0], (0.56, 0.87), "start")
    axs[0].set_ylabel("$x_2$")
    fig.tight_layout()
    save(fig, "fig_zigzag_scaling.pdf")


# ---------------------------------------------------------------- 7 local minima
def fig_local_minima():
    J = lambda x: x ** 4 - 6 * x ** 3 + 8 * x ** 2
    dJ = lambda x: 4 * x ** 3 - 18 * x ** 2 + 16 * x
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    plot_grid(ax)
    xx = np.linspace(-0.8, 4.2, 400)
    ax.plot(xx, J(xx), color=INK2, lw=1.6, zorder=2)
    xmax, xgm = (9 - np.sqrt(17)) / 4, (9 + np.sqrt(17)) / 4
    for x0, col in [(1.0, C1), (1.5, C2)]:
        P = gd(lambda x: dJ(x), x0, 0.05, 40)
        ax.plot(P, J(P), "o", color=col, ms=4.5, mec="white", mew=0.6, zorder=5)
        ax.plot(x0, J(x0), "s", color=col, ms=7, mec="white", mew=0.8, zorder=6)
    for x0, col, tx, ha in [(1.0, C1, 0.55, "right"), (1.5, C2, 1.85, "left")]:
        ax.annotate(f"start x$_0$ = {x0}", xy=(x0, J(x0)), xytext=(tx, 8.2), color=col, fontsize=8,
                    fontweight="bold", ha=ha, va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc=PAPER, ec="none"),
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8, shrinkA=2, shrinkB=5))
    mark_min(ax, (0, J(0)))
    mark_min(ax, (xgm, J(xgm)))
    ax.plot(xmax, J(xmax), "o", mfc="none", mec=BAD, mew=1.8, ms=9, zorder=7)
    tag(ax, 0.0, J(0) - 3.2, "local min", color=STAR, ha="center", va="top", fontweight="bold")
    tag(ax, xmax, J(xmax) + 2.0, "local max", color=BAD, ha="center", va="bottom", fontweight="bold")
    tag(ax, xgm, J(xgm) - 3.2, "global min", color=STAR, ha="center", va="top", fontweight="bold")
    ax.set_xlim(-0.8, 4.2)
    ax.set_ylim(J(xgm) - 8, 12)
    ax.set_xlabel("x")
    ax.set_ylabel("J(x)")
    save(fig, "fig_local_minima.pdf")


# ---------------------------------------------------------------- 8 GD vs Newton
def fig_gd_vs_newton():
    fp = lambda x: 2 * (x - 3) + 1.2 * (x - 3) ** 3
    fpp = lambda x: 2 + 3.6 * (x - 3) ** 2
    G = gd(fp, 0.0, 0.1, 60)
    x, Nw = 0.0, [0.0]
    for _ in range(12):
        x = x - fp(x) / fpp(x)
        Nw.append(x)
    Nw = np.array(Nw)
    fl = lambda v: np.maximum(np.abs(v - 3), 1e-16)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    plot_grid(ax)
    ax.semilogy(np.arange(len(G)), fl(G), "o-", color=C1, ms=3, lw=1.2, mec="white", mew=0.4)
    ax.semilogy(np.arange(len(Nw)), fl(Nw), "o-", color=C2, ms=4.5, lw=1.4, mec="white", mew=0.5, zorder=5)
    ax.axhline(1e-16, color=MUTED, lw=0.8, ls=":")
    tag(ax, 58, 4e-16, "machine floor", color=INK2, ha="right", va="bottom")
    callout(ax, (30, fl(G[30])), (0.50, 0.88),
            "GD: straight line\n= linear order", color=C1)
    callout(ax, (6, fl(Nw[6])), (0.24, 0.48),
            "Newton: digits double\n= quadratic order", color=C2)
    ax.set_xlim(-1, 60)
    ax.set_ylim(1e-17, 10)
    ax.set_xlabel("iteration k")
    ax.set_ylabel("$|x_k - 3|$")
    save(fig, "fig_gd_vs_newton.pdf")


# ---------------------------------------------------------------- 9 SGD paths
def fig_sgd_paths():
    A = np.column_stack([np.ones(3), X])
    opt = np.linalg.solve(A.T @ A, A.T @ Y)
    sgrad = lambda th, idx: -2 * A[idx].T @ (Y[idx] - A[idx] @ th)
    th0 = np.array([0.0, 0.0])
    rng = np.random.default_rng(0)
    idx_all = np.arange(3)
    runs = {}
    th, P = th0.copy(), [th0.copy()]
    for _ in range(300):
        th = th - 0.01 * sgrad(th, idx_all)
        P.append(th.copy())
    runs["batch GD"] = (np.array(P), C1)
    th, P = th0.copy(), [th0.copy()]
    for _ in range(300):
        th = th - 0.03 * sgrad(th, np.array([rng.integers(3)]))
        P.append(th.copy())
    runs["SGD"] = (np.array(P), C2)
    th, P = th0.copy(), [th0.copy()]
    for _ in range(300):
        th = th - 0.02 * sgrad(th, rng.choice(3, 2, replace=False))
        P.append(th.copy())
    runs["mini-batch (2)"] = (np.array(P), C3)

    fig = plt.figure(figsize=(7.2, 2.8))
    # Reserve a wider gap for the convergence panel's log ticks and SSR label.
    layout = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.7], wspace=0.36,
                              left=0.07, right=0.97, top=0.80, bottom=0.30)
    paths = layout[0].subgridspec(1, 3, wspace=0.18)
    axs = [fig.add_subplot(paths[0, i]) for i in range(3)]
    axs.append(fig.add_subplot(layout[1]))
    allp = np.vstack([r[0] for r in runs.values()])
    lo, hi = allp.min(0), allp.max(0)
    pad = 0.25 * (hi - lo)
    bl, bh = lo[0] - pad[0], hi[0] + pad[0]
    ml, mh = lo[1] - pad[1], hi[1] + pad[1]
    gb, gm = np.meshgrid(np.linspace(bl, bh, 300), np.linspace(ml, mh, 300))
    Z = ssr(gb, gm)
    smin = float(ssr(opt[0], opt[1]))
    lv = smin + np.geomspace(0.05, Z.max() - smin, 12)
    styles = ["-", "--", "-."]
    for ax, (name, (P, col)) in zip(axs[:3], runs.items()):
        contour_map(ax, gb, gm, Z, lv)
        ax.plot(*P.T, color=col, lw=0.9, alpha=0.85, zorder=4)
        ax.plot(*P[-1], "o", color=col, ms=4, mec="white", mew=0.6, zorder=5)
        mark_start(ax, th0)
        mark_min(ax, opt, ms=10)
        ax.set(xlim=(bl, bh), ylim=(ml, mh), xlabel="b")
        ax.set_title(name, color=col, fontsize=8, pad=9)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=8)
    axs[0].set_ylabel("m", fontsize=8)
    for ax in axs[1:3]:
        ax.tick_params(labelleft=False)
    a2 = axs[3]
    plot_grid(a2)
    floor = 1e-5
    for i, ((name, (P, col)), ls) in enumerate(zip(runs.items(), styles)):
        # Plot the gap to the best possible SSR, so the three methods separate.
        gap = np.maximum(ssr(P[:, 0], P[:, 1]) - smin, floor)
        a2.plot(gap, color=col, lw=0.5, alpha=0.18, zorder=2)
        smooth = np.exp(np.array([np.mean(np.log(gap[max(0, k - 14):k + 1]))
                                  for k in range(len(gap))]))
        a2.plot(smooth, color=col, lw=1.6, ls=ls, label=name, zorder=4)
    a2.set(yscale="log", ylim=(floor, 30), xlim=(0, 300),
           xlabel="iteration", ylabel=r"SSR $-$ SSR$^\star$")
    a2.set_title("gap to the best fit\n(15-step average)", fontsize=8, pad=9)
    a2.tick_params(labelsize=8)
    a2.set_xticks([0, 150, 300])
    a2.tick_params(axis="y", which="minor", labelleft=False)
    fig.legend(*a2.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.52, 0.015), ncol=3, frameon=False,
               fontsize=8, columnspacing=1.3, handlelength=2.2)
    save(fig, "fig_sgd_paths.pdf", tight=False)
    print("LS optimum (b, m) =", opt, "SSR* =", smin)


# ---------------------------------------------------------------- 10 can design
R_STAR = (250 / np.pi) ** (1 / 3)


def fig_can_design():
    A = lambda r: 2 * np.pi * r ** 2 + 1000 / r
    dA = lambda r: 4 * np.pi * r - 1000 / r ** 2
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    plot_grid(ax)
    rr = np.linspace(1.5, 8, 400)
    ax.plot(rr, A(rr), color=INK2, lw=1.7, zorder=2)
    P = gd(lambda r: dA(r), 2.0, 0.01, 20)
    print("can GD r_k:", np.round(P[:8], 3), "... last", P[-1])
    ax.plot(P, A(P), "o", color=C2, ms=4.5, mec="white", mew=0.6, zorder=5, clip_on=True)
    mark_start(ax, (2.0, A(2.0)))
    mark_min(ax, (R_STAR, A(R_STAR)))
    callout(ax, (R_STAR, A(R_STAR)), (0.56, 0.16),
            f"r* = {R_STAR:.2f} cm", color=STAR)
    callout(ax, (2.0, A(2.0)), (0.20, 0.77), "start r$_0$ = 2")
    ax.annotate("GD iterates: one big jump,\nthen crawling", xy=(P[1], A(P[1])), xytext=(5.0, 620),
                color=C2, fontsize=8, fontweight="bold", ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc=PAPER, ec="none"),
                arrowprops=dict(arrowstyle="-|>", color=C2, lw=0.9, shrinkA=2, shrinkB=4, mutation_scale=8,
                                connectionstyle="arc3,rad=0.25"))
    ax.set_xlim(1.5, 8)
    ax.set_ylim(200, 800)
    ax.set_xlabel("radius r (cm)")
    ax.set_ylabel("surface area A (cm²)")
    save(fig, "fig_can_design.pdf")


# ---------------------------------------------------------------- 11 logistic
def fig_logistic():
    x = np.arange(1, 6, dtype=float)
    y = np.array([0, 0, 1, 0, 1], float)
    th = np.zeros(2)
    snaps = {}
    for k in range(1, 20001):
        p = 1 / (1 + np.exp(-(th[0] * x + th[1])))
        r = p - y
        th = th - 0.5 * np.array([(r * x).mean(), r.mean()])
        if k in (1, 10, 100):
            snaps[k] = th.copy()
    w, b = th
    print("logistic final w, b =", w, b, " boundary", -b / w)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    plot_grid(ax)
    xx = np.linspace(0.5, 5.5, 300)
    sig = lambda t: 1 / (1 + np.exp(-(t[0] * xx + t[1])))
    for k in (1, 10, 100):
        ax.plot(xx, sig(snaps[k]), color=MUTED, lw=1.0, zorder=2)
    ax.plot(xx, sig(th), color=C1, lw=2.6, zorder=4)
    ax.axvline(-b / w, color=INK2, lw=1.0, ls=":", zorder=3)
    tag(ax, -b / w - 0.08, 0.8, f"P = 0.5 at {-b / w:.1f} h", color=INK2, ha="right")
    ax.plot(x, y, "o", color=INK, ms=7, mec="white", mew=0.8, zorder=6, clip_on=False)
    tag(ax, 0.04, 0.88, f"after 20000 steps:\nw = {w:.2f}, b = {b:.2f}",
        transform=ax.transAxes, color=C1, fontweight="bold", ha="left", va="top")
    tag(ax, 5.45, 0.2, "grey: after 1, 10,\n100 steps", color=INK2, ha="right", va="center", fontsize=8)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel("study hours")
    ax.set_ylabel("P(pass)")
    fig.subplots_adjust(right=0.93)
    save(fig, "fig_logistic.pdf")


# ---------------------------------------------------------------- 12 failure gallery
def fig_failure_gallery():
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2))

    def panel(ax, f, xx, P, ttl, ylim, xlim):
        plot_grid(ax)
        ax.plot(xx, f(xx), color=INK2, lw=1.3, zorder=2)
        ax.plot(P, f(P), "-", color=C2, lw=0.8, zorder=4, clip_on=True)
        ax.plot(P, f(P), "o", color=C2, ms=3.6, mec="white", mew=0.5, zorder=5, clip_on=True)
        ax.plot(P[0], f(P[0]), "s", color=INK, ms=5, mec="white", mew=0.6, zorder=6, clip_on=True)
        ax.set_title(ttl, color=BAD, fontsize=9, y=1.36, va="top", pad=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("x", fontsize=8)

    xx = np.linspace(-4, 4, 400)
    P = gd(lambda x: 2 * x, 1.5, 1.05, 9)
    panel(axs[0], lambda x: x ** 2, xx, P, "step too big", (-1, 16), (-4, 4))
    xx = np.linspace(-1, 1, 300)
    P = gd(lambda x: 3 * x ** 2, 0.6, 0.1, 14)
    panel(axs[1], lambda x: x ** 3, xx, P, "saddle:\nstuck", (-1.1, 1.1), (-1, 1))
    axs[1].set_xticks([-1, 0, 1])
    xx = np.linspace(-4, 4, 400)
    P = gd(lambda x: 2 * x * np.exp(-x ** 2), 2.5, 0.5, 8)
    panel(axs[2], lambda x: 1 - np.exp(-x ** 2), xx, P, "plateau:\ncrawling", (-0.15, 1.15), (-4, 4))
    xx = np.linspace(-1.2, 1.2, 300)
    P = [1.0]
    for _ in range(8):
        P.append(P[-1] - 0.3 * np.sign(P[-1]))
    panel(axs[3], np.abs, xx, np.array(P), "not smooth", (-0.1, 1.3), (-1.2, 1.2))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.24, top=0.72, wspace=0.44)
    save(fig, "fig_failure_gallery.pdf")


# ---------------------------------------------------------------- 13 stability window
def fig_stability_window():
    a = np.linspace(0, 0.32, 600)
    r1, r2 = np.abs(1 - 2 * a), np.abs(1 - 8 * a)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    plot_grid(ax)
    ax.axvspan(0, 0.25, color="#cdeae6", alpha=0.6, lw=0, zorder=0)
    ax.axhline(1, color=BAD, lw=1.0, ls="--", zorder=2)
    ax.plot(a, r1, color=C1, lw=1.5, zorder=3, label="slow direction ($\\lambda=2$)")
    ax.plot(a, r2, color=C2, lw=1.5, zorder=3, label="steep direction ($\\lambda=8$)")
    ax.plot(a, np.maximum(r1, r2), color=INK, lw=2.8, zorder=4,
            label="max = convergence factor $\\rho$")
    ax.axvline(0.25, color=STAR, lw=1.0, ls=":", zorder=2)
    ax.plot(0.2, 0.6, "*", color=STAR, ms=15, mec="white", mew=1.0, zorder=8)
    callout(ax, (0.2, 0.6), (0.48, 0.80),
            "$\\alpha^* = 0.2$, $\\rho^* = 0.6$", color=STAR)
    callout(ax, (0.25, 0.08), (0.61, 0.20), "$\\alpha = 2/8 = 0.25$", color=STAR)
    tag(ax, 0.012, 1.08, "unstable above this line ($\\rho > 1$)", color=BAD,
        ha="left", va="bottom")
    tag(ax, 0.015, 0.12, "stable region", color=STAR, ha="left",
        va="bottom", fontweight="bold")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.54, 0.015), fontsize=8, ncol=2, frameon=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.34)
    ax.set_xlim(0, 0.32)
    ax.set_ylim(0, 1.75)
    ax.set_xlabel("learning rate $\\alpha$")
    ax.set_ylabel("$|1 - \\alpha\\lambda|$")
    save(fig, "fig_stability_window.pdf")


# ---------------------------------------------------------------- 14 momentum
def fig_momentum():
    mu, L = 1.0, 100.0
    k = L / mu
    grad = lambda p: np.array([mu * p[0], L * p[1]])
    x0 = np.array([10.0, 1.0])
    sk = np.sqrt(k)

    def run(method, n):
        x, xp = x0.copy(), x0.copy()
        P = [x.copy()]
        for _ in range(n):
            if method == "gd":
                xn = x - 2 / (mu + L) * grad(x)
            elif method == "hb":
                xn = x - 4 / (np.sqrt(L) + np.sqrt(mu)) ** 2 * grad(x) + ((sk - 1) / (sk + 1)) ** 2 * (x - xp)
            else:
                y = x + (sk - 1) / (sk + 1) * (x - xp)
                xn = y - 1 / L * grad(y)
            xp, x = x, xn
            P.append(x.copy())
        return np.array(P)

    names = [("gd", "gradient descent", C1), ("hb", "heavy ball", C2), ("nag", "Nesterov", C3)]
    f = lambda X_, Y_: 0.5 * (mu * X_ ** 2 + L * Y_ ** 2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
    g1, g2 = np.meshgrid(np.linspace(-1.5, 10.8, 400), np.linspace(-1.5, 1.5, 300))
    contour_map(a1, g1, g2, f(g1, g2), [1, 3, 8, 20, 40, 70, 100, 150])
    for key, lab, col in names:
        P = run(key, 40)
        a1.plot(*P.T, "-", color=col, lw=1.0, zorder=4, alpha=0.9)
        a1.plot(*P.T, "o", color=col, ms=2.8, mec="white", mew=0.4, zorder=4.1, label=lab)
    mark_start(a1, x0)
    mark_min(a1, (0, 0), ms=12)
    a1.set_xlim(-1.5, 10.8)
    a1.set_ylim(-1.5, 1.5)
    a1.set_xlabel("$x_1$")
    a1.set_ylabel("$x_2$")
    a1.set_title("first 40 iterates", fontsize=9)
    plot_grid(a2)
    n0 = np.linalg.norm(x0)
    for key, lab, col in names:
        nrm = np.linalg.norm(run(key, 300), axis=1)
        a2.semilogy(nrm, color=col, lw=1.4, label=lab)
        nl = np.linalg.norm(run(key, 5000), axis=1)
        hit = int(np.argmax(nl <= 1e-6 * n0))
        print(f"momentum: {lab}: {hit} iterations to reach ||x_k|| <= 1e-6 ||x_0||")
    a2.axhline(1e-6 * n0, color=MUTED, lw=0.8, ls=":")
    a2.set_xlim(0, 300)
    a2.set_ylim(1e-12, 30)
    a2.set_xlabel("iteration k")
    a2.set_ylabel("$\\|x_k\\|$")
    a2.set_title("distance to minimiser", fontsize=9)
    fig.legend(*a2.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.53, 0.015), ncol=3, fontsize=8, frameon=False,
               handlelength=2.2, columnspacing=1.5)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.32, wspace=0.38)
    save(fig, "fig_momentum.pdf")


# ---------------------------------------------------------------- 15 rate bounds
def fig_rate_bounds():
    k = np.logspace(0, 3, 400)
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    plot_grid(ax)
    cur = [(k ** -0.5, INK2, "$1/\\sqrt{k}$: nonconvex"),
           (1 / k, C1, "$1/k$: convex GD"),
           (1 / k ** 2, C3, "$1/k^2$: convex, Nesterov"),
           (0.99 ** k, C2, "$(1-1/100)^k$: strongly convex GD"),
           (0.9 ** k, STAR, "$(1-1/10)^k$: accelerated")]
    for y, col, lab in cur:
        ax.loglog(k, y, color=col, lw=1.9, label=lab)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.52, 0.01), ncol=2, fontsize=8, frameon=False,
               columnspacing=1.3)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.40)
    ax.set_xlim(1, 1000)
    ax.set_ylim(1e-12, 8)
    ax.set_xlabel("iteration k")
    ax.set_ylabel("error bound")
    save(fig, "fig_rate_bounds.pdf")


# ---------------------------------------------------------------- 16 SGD noise
def fig_sgd_noise():
    xs, ys = np.array([1.0, 2.0, 3.0]), np.array([2.1, 3.9, 6.2])
    wst = 28.5 / 14
    nep, nseed = 200, 300
    rng = np.random.default_rng(1)
    modes = {"const": lambda k: 0.05, "decay": lambda k: 0.05 / (1 + 0.1 * k)}
    res = {}
    for m, sched in modes.items():
        w = np.zeros(nseed)
        err = np.zeros((nep + 1, nseed))
        err[0] = (w - wst) ** 2
        for ep in range(nep):
            a = sched(ep)
            perm = np.argsort(rng.random((nseed, 3)), axis=1)
            for j in range(3):
                i = perm[:, j]
                w = w - a * (w * xs[i] - ys[i]) * xs[i]
            err[ep + 1] = (w - wst) ** 2
        res[m] = err
        print(f"sgd_noise: {m}: final mean (w-w*)^2 over {nseed} seeds = {err[-1].mean():.3e}, "
              f"mean over last 50 epochs = {err[-50:].mean():.3e}")
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    plot_grid(ax)
    ep = np.arange(nep + 1)
    for m, col in [("const", C2), ("decay", C1)]:
        ax.semilogy(ep[1:], res[m][1:, 0], color=col, lw=0.6, alpha=0.4, zorder=2)
    for m, col in [("const", C2), ("decay", C1)]:
        ax.semilogy(ep[1:], res[m][1:].mean(1), color=col, lw=2.2, zorder=4,
                    label=("constant $\\alpha$: noise floor" if m == "const" else
                           "decaying $\\alpha_k = 0.05/(1+0.1k)$"))
    floor = res["const"][-50:].mean()
    ax.axhline(floor, color=C2, lw=0.8, ls=":", zorder=3)
    ax.set_title("Thick: mean of 300 runs; thin: one run", fontsize=9)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.53, 0.01), fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.87, bottom=0.34)
    ax.set_xlim(0, 200)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean $(w_k - w^*)^2$")
    ax.set_ylim(top=3)
    save(fig, "fig_sgd_noise.pdf")


# ---------------------------------------------------------------- 17 Rosenbrock race
def fig_rosenbrock_race():
    f = lambda x, y: (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    grad = lambda p: np.array([-2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] ** 2), 200 * (p[1] - p[0] ** 2)])
    x0 = np.array([-1.2, 1.0])
    n = 5000
    a_gd, a_hb, a_adam = 1e-3, 5e-4, 0.02
    P = {}
    x = x0.copy()
    P["gd"] = [x.copy()]
    for _ in range(n):
        x = x - a_gd * grad(x)
        P["gd"].append(x.copy())
    x, v = x0.copy(), np.zeros(2)
    P["hb"] = [x.copy()]
    for _ in range(n):
        v = 0.9 * v - a_hb * grad(x)
        x = x + v
        P["hb"].append(x.copy())
    x, m, v = x0.copy(), np.zeros(2), np.zeros(2)
    P["adam"] = [x.copy()]
    for t in range(1, n + 1):
        g = grad(x)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        x = x - a_adam * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        P["adam"].append(x.copy())
    P = {k_: np.array(v_) for k_, v_ in P.items()}
    names = [("gd", f"GD ($\\alpha$={a_gd:g})", C1), ("hb", f"momentum ($\\alpha$={a_hb:g})", C2),
             ("adam", f"Adam ($\\alpha$={a_adam:g})", C3)]
    print(f"rosenbrock: step sizes GD={a_gd}, heavy ball={a_hb} (beta 0.9), Adam={a_adam}")
    for k_, lab, _ in names:
        print(f"rosenbrock: {k_}: final f = {f(*P[k_][-1]):.3e}, final point = {P[k_][-1]}")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 3.0), gridspec_kw=dict(width_ratios=[1.1, 1]))
    g1, g2 = np.meshgrid(np.linspace(-1.6, 1.6, 400), np.linspace(-0.6, 1.8, 400))
    contour_map(a1, g1, g2, f(g1, g2), np.geomspace(0.05, 2500, 16))
    for (k_, lab, col), lw_, z_ in zip(names, [3.2, 2.0, 1.0], [4, 4.1, 4.2]):
        a1.plot(*P[k_][::5].T, "-", color=col, lw=lw_, zorder=z_, alpha=0.95, label=lab)
    mark_start(a1, x0)
    mark_min(a1, (1, 1), ms=13)
    a1.set_xlim(-1.6, 1.6)
    a1.set_ylim(-0.6, 1.8)
    a1.set_xlabel("x")
    a1.set_ylabel("y")
    a1.set_title("paths, 5000 steps", fontsize=9)
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center",
               bbox_to_anchor=(0.53, 0.015), ncol=3, fontsize=8, frameon=False,
               handlelength=2.0, columnspacing=1.2)
    plot_grid(a2)
    fv = {k_: np.maximum(f(P[k_][:, 0], P[k_][:, 1]), 1e-30) for k_ in P}
    for k_, lab, col in names:
        a2.semilogy(fv[k_], color=col, lw=1.3)
    a2.set_xlim(0, n)
    a2.set_ylim(1e-14, 1e3)
    a2.set_xlabel("iteration k")
    a2.set_ylabel("$f(x_k)$")
    a2.set_title("objective value", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.32, wspace=0.38)
    save(fig, "fig_rosenbrock_race.pdf")


def print_condition_numbers():
    def kap(H):
        e = np.linalg.eigvalsh(H)
        return e.max() / e.min(), e
    print("kappa (a) J=x1^2+4x2^2:", kap(np.diag([2.0, 8.0])))
    size, beds = np.array([800, 1000, 1200, 1500.0]), np.array([2, 2, 3, 3.0])
    Xh = np.column_stack([np.ones(4), size, beds])
    print("kappa (b) house raw X^T X/4:", kap(Xh.T @ Xh / 4))
    Z = np.column_stack([np.ones(4), (size - size.mean()) / size.std(), (beds - beds.mean()) / beds.std()])
    print("kappa (c) house z-scored X^T X/4:", kap(Z.T @ Z / 4))
    x = np.array([0.5, 2.3, 2.9])
    A = np.column_stack([np.ones(3), x])
    print("kappa (d) weight-height raw X^T X:", kap(A.T @ A))
    A = np.column_stack([np.ones(3), x - x.mean()])
    print("kappa (d) weight-height centred X^T X:", kap(A.T @ A))
    print("kappa (e) Rosenbrock Hessian at (1,1):", kap(np.array([[802.0, -400.0], [-400.0, 200.0]])))


# ---------------------------------------------------------------- story figures
# GD on SSR(b, m) for the weight-height data (Problem A1 settings)
A1_START, A1_ALPHA, A1_STEPS = np.array([0.0, 1.0]), 0.01, 808


def ssr_grad(p):
    r = Y - p[0] - p[1] * X
    return np.array([-2 * r.sum(), -2 * (X * r).sum()])


A1_PATH = gd(ssr_grad, A1_START, A1_ALPHA, A1_STEPS)
_A = np.column_stack([np.ones(3), X])
A1_OPT = np.linalg.solve(_A.T @ _A, _A.T @ Y)
BL, BH, ML, MH = -0.6, 2.2, -0.15, 1.35


def _pane_style(ax):
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a.pane.set_edgecolor(GRIDC)
        a._axinfo["grid"]["color"] = GRIDC
        a._axinfo["grid"]["linewidth"] = 0.5


def fig_ssr_surface():
    fig = plt.figure(figsize=(5.6, 4.2))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    fig.subplots_adjust(left=-0.06, right=1.0, bottom=-0.02, top=1.06)
    gb, gm = np.meshgrid(np.linspace(-0.2, 1.8, 150), np.linspace(0.15, 1.2, 110))
    Z = ssr(gb, gm)
    floor = -5.0
    lv = float(Z.min()) + np.geomspace(0.03, float(Z.max() - Z.min()), 14)
    ax.contourf(gb, gm, Z, levels=np.r_[Z.min() - 1e-9, lv], zdir="z", offset=floor, cmap=SEQ,
                alpha=0.9, zorder=0)
    ax.contour(gb, gm, Z, levels=lv, zdir="z", offset=floor, colors="#5b7fae", linewidths=0.4,
               alpha=0.7, zorder=1)
    ls = LightSource(azdeg=300, altdeg=50)
    rgb = ls.shade(Z, cmap=SURF, vert_exag=0.5, blend_mode="soft", vmin=0, vmax=16)
    ax.plot_surface(gb, gm, Z, facecolors=rgb, linewidth=0, antialiased=True, rstride=2, cstride=2,
                    shade=False, rasterized=True, alpha=0.78, zorder=2)
    P = A1_PATH
    zp = ssr(P[:, 0], P[:, 1]) + 0.25
    ax.plot(P[:, 0], P[:, 1], np.full(len(P), floor), color=C2, lw=1.4, ls="--", zorder=3)
    ax.plot(P[:, 0], P[:, 1], zp, color=C2, lw=2.4, zorder=10)
    sel = np.r_[0:4, 8:len(P):80]
    ax.scatter(P[sel, 0], P[sel, 1], zp[sel], color=C2, s=14, edgecolor="white", linewidths=0.5,
               depthshade=False, zorder=11)
    ax.scatter([P[0, 0]], [P[0, 1]], [zp[0]], marker="s", color=INK, s=36, edgecolor="white",
               depthshade=False, zorder=12)
    zs = float(ssr(*A1_OPT)) + 0.25
    ax.scatter([A1_OPT[0]], [A1_OPT[1]], [zs], marker="*", color=STAR, s=210, edgecolor="white",
               linewidths=0.8, depthshade=False, zorder=12)
    ax.scatter([A1_OPT[0]], [A1_OPT[1]], [floor], marker="*", color=STAR, s=120, edgecolor="white",
               linewidths=0.6, depthshade=False, zorder=4)
    ax.text(P[0, 0] - 0.05, P[0, 1] + 0.05, zp[0] + 2.2, "start", color=INK, fontsize=8, zorder=13)
    ax.text(A1_OPT[0] + 0.32, A1_OPT[1] - 0.05, zs + 1.2, "best line", color=STAR, fontsize=8,
            fontweight="bold", zorder=13)
    _pane_style(ax)
    ax.set_zlim(floor, 15)
    ax.set_zticks([0, 5, 10, 15])
    ax.set_xlabel("intercept b", labelpad=-2)
    ax.set_ylabel("slope m", labelpad=-2)
    ax.set_zlabel("SSR", labelpad=-4)
    ax.tick_params(labelsize=7, pad=-2)
    ax.view_init(elev=27, azim=-62)
    save(fig, "fig_ssr_surface.pdf", tight=False)


FRAME_K = [0, 1, 2, 4, 15, 60, 250, 808]


def fig_ssr_frames():
    gb, gm = np.meshgrid(np.linspace(BL, BH, 300), np.linspace(ML, MH, 300))
    Z = ssr(gb, gm)
    smin = float(ssr(*A1_OPT))
    lv = smin + np.geomspace(0.02, Z.max() - smin, 16)
    xx = np.linspace(0, 3.2, 50)
    for i, k in enumerate(FRAME_K):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                                     gridspec_kw=dict(width_ratios=[1.15, 1]))
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.88, wspace=0.28)
        contour_map(a1, gb, gm, Z, lv)
        P = A1_PATH[: k + 1]
        a1.plot(*P.T, "-", color=INK2, lw=1.2, zorder=4)
        a1.plot(*A1_PATH[[j for j in range(min(k, 5) + 1)]].T, "o", color=INK2, ms=2.6, zorder=4)
        mark_start(a1, A1_START)
        mark_min(a1, A1_OPT, ms=12)
        p = A1_PATH[k]
        a1.plot(*p, "o", color=PATH, ms=8, mec="white", mew=1.2, zorder=8)
        g = ssr_grad(p)
        if np.linalg.norm(g) > 1e-3:
            d = -g / np.linalg.norm(g) * 0.35
            a1.annotate("", xy=p + d, xytext=p, zorder=9,
                        arrowprops=dict(arrowstyle="-|>", color=PATH, lw=2.0, mutation_scale=12))
        a1.set(xlim=(BL, BH), ylim=(ML, MH), xlabel="intercept b", ylabel="slope m")
        a1.set_title(r"parameters, with $-\nabla$SSR (orange)", fontsize=8.5, color=INK2,
                     fontweight="normal")
        plot_grid(a2)
        a2.plot(X, Y, "o", color=INK, ms=6, zorder=5)
        yh = p[0] + p[1] * X
        for xi, yi, hi in zip(X, Y, yh):
            a2.plot([xi, xi], [yi, hi], color=BAD, lw=1.1, ls="--", zorder=4)
        a2.plot(xx, p[0] + p[1] * xx, color=PATH, lw=2.0, zorder=3)
        a2.set(xlim=(0, 3.2), ylim=(0, 4), xlabel="weight", ylabel="height")
        a2.set_title("the line those parameters draw", fontsize=8.5, color=INK2, fontweight="normal")
        tag(a2, 0.04, 0.9, f"step {k}:  b = {p[0]:.3f},  m = {p[1]:.3f},  SSR = {float(ssr(*p)):.3f}",
            fontsize=8, transform=a2.transAxes)
        save(fig, f"fig_ssr_frame_{i}.pdf", tight=False)
    print("SSR frames at k =", FRAME_K, "end =", A1_PATH[-1], "opt =", A1_OPT)


def f_quartic(x, y):
    return x ** 4 + y ** 4 - 4 * x * y


def fig_quartic_surface():
    fig = plt.figure(figsize=(5.0, 3.8))
    ax = fig.add_subplot(projection="3d")
    U, V = np.meshgrid(np.linspace(-2.1, 2.1, 160), np.linspace(-0.7, 0.7, 80))
    Xq, Yq = (U + V) / np.sqrt(2), (U - V) / np.sqrt(2)
    Z = f_quartic(Xq, Yq)
    ls = LightSource(azdeg=315, altdeg=40)
    rgb = ls.shade(Z, cmap=SURF, vert_exag=0.6, blend_mode="soft", vmin=-2, vmax=7)
    ax.plot_surface(Xq, Yq, Z, facecolors=rgb, linewidth=0, antialiased=True, rstride=1, cstride=1,
                    shade=False, rasterized=True)
    ax.scatter([0], [0], [0.25], marker="X", color=BAD, s=90, edgecolor="white", linewidths=0.8,
               depthshade=False, zorder=10)
    ax.scatter([1, -1], [1, -1], [-1.75, -1.75], marker="*", color=STAR, s=180, edgecolor="white",
               linewidths=0.8, depthshade=False, zorder=10)
    ax.text(0.15, -0.2, 1.6, "saddle", color=BAD, fontsize=8, fontweight="bold", zorder=11)
    ax.text(1.0, 1.25, -0.6, "minimum", color=STAR, fontsize=8, fontweight="bold", zorder=11)
    ax.text(-1.4, -1.3, -0.4, "minimum", color=STAR, fontsize=8, fontweight="bold", zorder=11)
    ax.set_zlim(-2.6, 6.5)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_zticks([-2, 0, 2, 4, 6])
    _pane_style(ax)
    ax.set_xlabel("$x_1$", labelpad=-4)
    ax.set_ylabel("$x_2$", labelpad=-4)
    ax.set_zlabel("J", labelpad=-6)
    ax.tick_params(labelsize=7, pad=-2)
    ax.view_init(elev=26, azim=-58)
    save(fig, "fig_quartic_surface.pdf")


def fig_quartic_contour():
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    gx, gy = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-2, 2, 300))
    Z = f_quartic(gx, gy)
    lv = -2 + np.geomspace(0.05, Z.max() + 2, 26)
    contour_map(ax, gx, gy, Z, lv)
    qx, qy = np.meshgrid(np.linspace(-1.85, 1.85, 13), np.linspace(-1.85, 1.85, 13))
    Gx, Gy = 4 * qx ** 3 - 4 * qy, 4 * qy ** 3 - 4 * qx
    n = np.hypot(Gx, Gy) + 1e-12
    ax.quiver(qx, qy, -Gx / n, -Gy / n, color=INK2, scale=26, width=0.0045, headwidth=4,
              headlength=4.5, alpha=0.85, zorder=3)
    ax.plot(0, 0, "X", color=BAD, ms=11, mec="white", mew=1.0, zorder=6)
    mark_min(ax, (1, 1))
    mark_min(ax, (-1, -1))
    tag(ax, 0.14, -0.32, "saddle, J = 0", color=BAD)
    tag(ax, 0.55, 1.35, "min, J = −2", color=STAR)
    tag(ax, -1.6, -1.55, "min, J = −2", color=STAR)
    ax.set_aspect("equal")
    ax.set(xlabel="$x_1$", ylabel="$x_2$")
    ax.set_title(r"arrows: $-\nabla J$", fontsize=9, color=INK2, fontweight="normal")
    save(fig, "fig_quartic_contour.pdf")


if __name__ == "__main__":
    # Only the line-fit figures used in this deck (the rest come from make_figures.py).
    for fn in [fig_line_frames, fig_local_minima, fig_sgd_paths, fig_ssr_surface, fig_ssr_frames]:
        fn()
