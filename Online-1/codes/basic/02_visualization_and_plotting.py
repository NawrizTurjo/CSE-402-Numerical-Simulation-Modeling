"""
=============================================================================
  TOPIC 2: Visualization & Plotting with Matplotlib
=============================================================================

KEY CONCEPTS:
  - Plot a function to FIND intervals where roots exist (graphical method)
  - Incremental search: scan with small Δx to detect sign changes
  - Show iteration convergence on a semi-log axis
  - Teachers EXPLICITLY mark on plotting quality - use these patterns!

EXAM PATTERNS FROM PREVIOUS BATCHES:
  1. "Plot f(x) on [a, b]" → plt.plot, add axhline(0), grid, labels, title
  2. "Use log scale" → plt.xscale('log') or plt.yscale('log')
  3. "Show sign-change intervals" → scatter the bracket endpoints
  4. "Show convergence" → semilogy(iterations, errors)
  5. "Shade the target interval" → plt.axvspan(xl, xu, alpha=0.3)
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Incremental search to detect sign-change intervals
# ─────────────────────────────────────────────────────────────────────────────

def find_sign_change_intervals(f, a, b, step=0.1):
    """
    Scan [a, b] with step size h. Where f flips sign, a root is bracketed.
    Returns a list of (xl, xu) tuples.
    EXAM TIP: Always do this BEFORE bisection/false-position to verify bracket.
    """
    intervals = []
    xs = np.arange(a, b, step)
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i + 1]
        if f(x1) * f(x2) < 0:      # sign change detected
            intervals.append((round(x1, 10), round(x2, 10)))
    return intervals


# ─────────────────────────────────────────────────────────────────────────────
# PLOT TEMPLATE 1 - Standard function plot with x-axis and grid
# ─────────────────────────────────────────────────────────────────────────────

def plot_function(f, a, b, title, fname=None, n=1000):
    """
    The most common exam plot. Shows the curve, zero line, labels, grid.
    Copy this template and change f, a, b, title for any exam question.
    """
    x = np.linspace(a, b, n)
    y = f(x)                         # NumPy will vectorize element-wise

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color='steelblue', linewidth=2, label='f(x)')
    plt.axhline(0, color='black', linewidth=1, linestyle='--')   # y = 0 line
    plt.axvline(0, color='gray',  linewidth=0.8, linestyle=':')  # y-axis

    plt.xlabel('x',    fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title(title,   fontsize=13)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT TEMPLATE 2 - Log scale (used in the previous year ln(x) question)
# ─────────────────────────────────────────────────────────────────────────────

def plot_function_logscale(f, a, b, title, fname=None):
    """
    Used when x-range spans many orders of magnitude (e.g., 1e-4 to 1e4).
    plt.xscale('log') makes equal spacing in decade units.
    """
    x = np.logspace(np.log10(a), np.log10(b), 1000)  # log-spaced points
    y = f(x)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color='darkgreen', linewidth=2, label='f(x)')
    plt.axhline(0, color='red', linewidth=1, linestyle='--', label='y = 0')
    plt.xscale('log')                   # THIS makes it log scale on x-axis

    plt.xlabel('x (log scale)', fontsize=12)
    plt.ylabel('f(x)',          fontsize=12)
    plt.title(title,            fontsize=13)
    plt.legend()
    plt.grid(True, which='both', alpha=0.4)   # 'both' = major + minor grid ticks
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT TEMPLATE 3 - Mark sign-change brackets + root on function plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_with_brackets(f, a, b, intervals, roots=None, title='', fname=None):
    """
    Overlays coloured vertical spans for bracketed root intervals, and
    marks the converged roots with red X markers.
    """
    x = np.linspace(a, b, 1000)
    y = f(x)

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, color='steelblue', linewidth=2, label='f(x)')
    plt.axhline(0, color='black', linewidth=1)

    colors = ['orange', 'green', 'violet', 'cyan']
    for k, (xl, xu) in enumerate(intervals):
        # shade the bracketed interval
        plt.axvspan(xl, xu, color=colors[k % len(colors)], alpha=0.35,
                    label=f'Bracket [{xl:.2f}, {xu:.2f}]')
        # mark the two bracket endpoints
        plt.scatter([xl, xu], [f(xl), f(xu)], color='red', s=60, zorder=5)

    if roots:
        # mark converged roots with x
        plt.scatter(roots, [0] * len(roots), color='red', marker='x',
                    s=120, linewidths=2.5, zorder=6, label='Root(s)')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT TEMPLATE 4 - Convergence plot (semi-log axis)
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(errors, title='Convergence of ea', fname=None):
    """
    Shows how the approximate relative error decreases per iteration.
    Linear decay on log-y → linear convergence (bisection / false position).
    Rapid steepening at the bottom → quadratic convergence (Newton-Raphson).
    """
    iters = list(range(1, len(errors) + 1))

    plt.figure(figsize=(7, 4))
    plt.semilogy(iters, errors, 'o-', color='darkred', markersize=5)
    plt.xlabel('Iteration number', fontsize=12)
    plt.ylabel('ea (%) [log scale]', fontsize=12)
    plt.title(title, fontsize=13)
    plt.grid(alpha=0.4, which='both')
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# DEMO - Run all templates on example functions
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Example 1: ln(x) from previous year
    f_ln = lambda x: np.log(x)
    plot_function_logscale(f_ln, 1e-4, 1e4,
                           title='f(x) = ln(x) - Log Scale Plot',
                           fname='plot_ln_logscale.png')

    # Example 2: multi-root function from section B
    sth = 1.0   # ← change this to match exam coefficient
    f_sec_b = lambda x: 0.6 * np.log(x + 1) - sth * np.sin(1.7 * x) - 0.08 * x**2 - 0.08
    plot_function(f_sec_b, 0, 10,
                  title=f'f(x) = 0.6 ln(x+1) - {sth}*sin(1.7x) - 0.08x^2 - 0.08',
                  fname='plot_section_b.png')

    # Find and mark brackets
    intervals = find_sign_change_intervals(f_sec_b, 0, 10, step=0.1)
    print(f"Sign-change intervals (step=0.1): {[(round(a,2), round(b,2)) for a,b in intervals]}")
    plot_with_brackets(f_sec_b, 0, 10, intervals,
                       title='Sign-Change Brackets for f(x)',
                       fname='plot_brackets.png')

    # Example 3: Chapra polynomial (3 roots)
    f_poly = lambda x: 2*x**3 - 11.7*x**2 + 17.7*x - 5
    plot_function(f_poly, 0, 4,
                  title='f(x) = 2x^3 - 11.7x^2 + 17.7x - 5',
                  fname='plot_chapra.png')

    # Example 4: fake convergence plot (replace with real ea values from solver)
    fake_errors = [100.0, 45.2, 21.0, 9.8, 4.5, 2.1, 1.0, 0.4, 0.18, 0.08, 0.03]
    plot_convergence(fake_errors, title='Bisection Convergence Demo',
                     fname='plot_convergence_demo.png')

    print("All plots saved.")
