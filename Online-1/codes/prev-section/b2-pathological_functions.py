"""
Section-B2: Pathological/Discontinuous Functions (Bisection) + Root Candidate Classification
Objective:
  f(x) = ((x-2.5)^2 * (x+1.052)) / (x-3.551)

Tasks (exact exam spec):
  1. Plot the function over -2 <= x <= 5.
  2. Scan with step=0.1: |f(x)| <= 1e-8 -> direct root candidate;
     sign change between grid points -> candidate interval.
  3. Run Bisection on every candidate interval, |ea| <= 0.0001%.
  4. If an undefined value is hit during a bisection run, terminate that run.
  5. Accept an estimate only if |x_r| <= 1e-6.
  6. Print iteration tables for every bisection run.
  7. Print candidate roots: type (direct/interval), functional value, accept/reject verdict.
  8. Explain (3-4 sentences): no-sign-change root, no-root sign-change interval, why check residuals.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
X_MIN = -2.0
X_MAX = 5.0
STEP = 0.1
TOL = 0.0001        # |ea| stopping tolerance (%)
DIRECT_EPS = 1e-8   # |f(x)| <= this -> direct root candidate
ACCEPT_EPS = 1e-6   # |x_r| <= this -> accepted estimate (exact exam wording)
MAX_ITER = 200

# -----------------------------------------------------------------------------
# Function Definition
# -----------------------------------------------------------------------------
def f(x):
    """Rational pathological function. Raises on the exact singularity."""
    denom = x - 3.551
    if abs(denom) < 1e-15:
        raise ZeroDivisionError("undefined at asymptote x = 3.551")
    return ((x - 2.5) ** 2 * (x + 1.052)) / denom

def f_safe(x):
    """f(x) but returns None instead of raising, for scan/bisection guards."""
    try:
        val = f(x)
    except ZeroDivisionError:
        return None
    if not math.isfinite(val):
        return None
    return val

def f_np(x):
    """Numpy vectorized version for plotting, with masking near the asymptote."""
    denom = x - 3.551
    with np.errstate(divide='ignore', invalid='ignore'):
        y = ((x - 2.5) ** 2 * (x + 1.052)) / denom
    y[np.abs(denom) < 0.02] = np.nan
    y[np.abs(y) > 100] = np.nan
    return y

# -----------------------------------------------------------------------------
# Task 2: Candidate Scan (direct roots + sign-change intervals)
# -----------------------------------------------------------------------------
def scan_candidates(x_start, x_end, step_size):
    xs = np.arange(x_start, x_end + step_size, step_size)
    direct_candidates = []
    interval_candidates = []

    vals = [f_safe(x) for x in xs]

    for i, v in enumerate(vals):
        if v is not None and abs(v) <= DIRECT_EPS:
            direct_candidates.append(xs[i])

    for i in range(len(xs) - 1):
        v1, v2 = vals[i], vals[i + 1]
        if v1 is None or v2 is None:
            continue
        if v1 * v2 < 0:
            interval_candidates.append((xs[i], xs[i + 1]))

    return direct_candidates, interval_candidates

# -----------------------------------------------------------------------------
# Task 3 & 4: Bisection Solver with Undefined-Value Termination
# -----------------------------------------------------------------------------
def run_bisection(xl, xu, root_index):
    """Run Bisection on bracket [xl, xu]. Terminates immediately if an
    undefined function value is encountered anywhere in the run."""
    print(f"\n[INTERVAL CANDIDATE #{root_index}] Bisection on [{xl:.4f}, {xu:.4f}]")
    print("-" * 72)
    print(f"{'Iter':<6} {'xl':>12} {'xu':>12} {'xr':>12} {'ea (%)':>12} {'f(xr)':>14}")
    print("-" * 72)

    xr_old = None
    xr = None
    converged = False
    terminated = False

    for i in range(1, MAX_ITER + 1):
        fl = f_safe(xl)
        if fl is None:
            print(f"  [TERMINATED] Undefined f(xl) at xl = {xl:.6f}.")
            terminated = True
            break

        xr = (xl + xu) / 2.0
        fxr = f_safe(xr)
        if fxr is None:
            print(f"{i:<6} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {'---':>12} {'undefined':>14}")
            print(f"  [TERMINATED] Undefined f(xr) at xr = {xr:.6f}.")
            terminated = True
            break

        if xr_old is not None:
            # ea = abs((xr - xr_old) / xr) * 100.0 if xr != 0 else float('inf')  # Alternative exact-zero infinity sentinel
            ea = abs((xr - xr_old) / xr) * 100.0 if abs(xr) >= 1e-15 else abs(xr - xr_old) * 100.0
            ea_str = f"{ea:.6f}"
        else:
            ea = 100.0
            ea_str = "---"

        print(f"{i:<6} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str:>12} {fxr:>14.6e}")

        if xr_old is not None and ea <= TOL:
            converged = True
            break

        if fl * fxr < 0:
            xu = xr
        else:
            xl = xr

        xr_old = xr

    print("-" * 72)
    if terminated:
        print("Result: Run terminated due to undefined value. No estimate produced.")
        return None
    if converged:
        print(f"Result: Converged to x approx. {xr:.8f} (iterations: {i}, ea = {ea:.6f}%)")
    else:
        print(f"Result: Ended at x approx. {xr:.8f} without meeting tolerance (iter cap hit).")
    return xr

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("      SECTION-B2: PATHOLOGICAL FUNCTION + ROOT CANDIDATE CLASSIFIER")
    print("=" * 80)

    # Task 2: scan for direct roots + sign-change intervals
    direct_candidates, interval_candidates = scan_candidates(X_MIN, X_MAX, STEP)

    print(f"\nCandidate scan (step = {STEP}):")
    print(f"  Direct root candidates (|f(x)| <= {DIRECT_EPS}): "
          f"{[round(x, 4) for x in direct_candidates] if direct_candidates else 'none'}")
    print(f"  Sign-change interval candidates: "
          f"{[(round(a,2), round(b,2)) for a, b in interval_candidates] if interval_candidates else 'none'}")

    # Task 3 & 4: run bisection on every interval candidate
    interval_results = []
    for idx, (xl, xu) in enumerate(interval_candidates, 1):
        xr = run_bisection(xl, xu, idx)
        interval_results.append(((xl, xu), xr))

    # Task 5 & 7: build candidate verdict table (direct + interval, in one place)
    print("\n" + "=" * 90)
    print("                         CANDIDATE ROOT VERDICT TABLE")
    print("=" * 90)
    header = f"{'Type':<10} {'Candidate':>16} {'f(value)':>16} {'Verdict':>12}"
    print(header)
    print("-" * 90)

    rows = []
    for x0 in direct_candidates:
        fval = f_safe(x0)
        verdict = "ACCEPT" if abs(x0) <= ACCEPT_EPS else "REJECT"
        rows.append(("direct", f"{x0:.6f}", fval, verdict))

    for (xl, xu), xr in interval_results:
        label = f"[{xl:.2f},{xu:.2f}]"
        if xr is None:
            rows.append(("interval", label, None, "REJECT (terminated)"))
        else:
            fval = f_safe(xr)
            verdict = "ACCEPT" if abs(xr) <= ACCEPT_EPS else "REJECT"
            rows.append(("interval", f"{label} -> {xr:.6f}", fval, verdict))

    for typ, cand, fval, verdict in rows:
        fval_str = f"{fval:.6e}" if fval is not None else "undefined"
        print(f"{typ:<10} {cand:>16} {fval_str:>16} {verdict:>12}")
    if not rows:
        print("(no candidates found)")
    print("=" * 90)

    # Task 1: Plot
    x_plot = np.linspace(X_MIN, X_MAX, 2000)
    y_plot = f_np(x_plot)

    plt.figure(figsize=(10, 6.5))
    plt.plot(x_plot, y_plot, label=r"$f(x) = \frac{(x-2.5)^2(x+1.052)}{x-3.551}$", color="teal", linewidth=2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
    plt.axvline(3.551, color="red", linestyle=":", linewidth=2, label="Vertical Asymptote (x = 3.551)")
    plt.plot(2.5, 0, "bo", markersize=8, label="Double Root (x = 2.5, missed by sign scan)")
    plt.plot(-1.052, 0, "go", markersize=8, label="Simple Root (x = -1.052)")

    for (xl, xu), xr in interval_results:
        if xr is not None:
            fv = f_safe(xr)
            if fv is not None:
                plt.plot(xr, fv, "ro", markersize=7)

    plt.title("Section-B2: Rational & Pathological Root Behavior", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("x", fontsize=11)
    plt.ylabel("f(x)", fontsize=11)
    plt.ylim(-30, 30)
    plt.xlim(X_MIN, X_MAX)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=9, loc="lower left")
    plt.tight_layout()

    plot_path = "b2_pathological_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved plot to '{plot_path}'")

    # Task 8: Written conceptual analysis
    print("""
--------------------------------------------------------------------------------
CONCEPTUAL ANALYSIS (Task 8):
1. The root at x = 2.5 has even multiplicity, factor (x-2.5)^2 >= 0 always, so
   f(x) keeps the same sign on both sides of it and never crosses zero there.
2. The interval scanned around x = 3.551 shows a sign change only because the
   denominator flips sign across that point (-inf to +inf jump), not because a
   root exists there - it is a discontinuity, not a zero of f.
3. We must check residuals / apply the |x_r| <= 1e-6 acceptance filter because
   a converged bisection estimate can still be a garbage root (asymptote) or an
   estimate that satisfies the ea-tolerance purely from interval shrinkage,
   without f(x_r) actually being close to zero or being the value we asked for.
--------------------------------------------------------------------------------
""")

if __name__ == "__main__":
    main()
