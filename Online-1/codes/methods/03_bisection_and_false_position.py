"""
=============================================================================
  TOPIC 3: Bisection Method & False Position (Regula Falsi) Method
=============================================================================

HOW TO USE IN THE EXAM:
  1. Redefine f(x) below to match the question (all functions in this file
     read that same global f - none of them take f as a parameter).
  2. Copy whichever function(s) you need into your answer script.
  3. Run.

ALGORITHM - BISECTION:
  Given bracket [xl, xu] where f(xl)*f(xu) < 0:
    1. xr = (xl + xu) / 2
    2. ea = |xr_new - xr_old| / |xr_new| x 100
    3. If f(xl)*f(xr) < 0 -> xu = xr   (root in left half)
       Else                -> xl = xr   (root in right half)
    4. Repeat until ea <= tolerance

ALGORITHM - FALSE POSITION (Regula Falsi):
  Given bracket [xl, xu] where f(xl)*f(xu) < 0:
    1. xr = xu - f(xu) x (xl - xu) / (f(xl) - f(xu))
         <- equivalent form: xr = (xu*f(xl) - xl*f(xu)) / (f(xl) - f(xu))
    2. ea = |xr_new - xr_old| / |xr_new| x 100
    3. Update bracket same as bisection
    4. Repeat until ea <= tolerance

DIFFERENCES:
  Bisection    -> always splits interval in half; slow but guaranteed
  False Pos    -> uses linear interpolation; faster for smooth functions,
                 BUT can be SLOWER if one bound never updates (stagnation)

KEY EXAM OUTPUTS EXPECTED:
  Iter table: | Iter | xl | xu | xr | ea(%) | f(xr) |
  Comparison table: both methods side by side
  Plot: function + convergence
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# (0) DEFINE YOUR FUNCTION HERE - change this for every question.
#     Every solver below reads this same global f(x) (no f parameter).
# -----------------------------------------------------------------------------

def f(x):
    return x**3 - x - 1


# -----------------------------------------------------------------------------
# CANDIDATE SCANNING & ACCEPTANCE FILTERING
# (needed for pathological/discontinuous-function questions, e.g. exam B2)
# -----------------------------------------------------------------------------

def safe_eval(fn, x):
    """
    Evaluate fn(x), catching domain errors (division by zero, log of a
    non-positive number, overflow) instead of letting them crash the caller.
    Returns (value, None) on success, (None, error_message) on failure.
    """
    try:
        return fn(x), None
    except (ZeroDivisionError, ValueError, OverflowError) as e:
        return None, str(e)


def coarse_scan(fn, a, b, step=0.1, direct_tol=1e-8):
    """
    Scan [a, b] with a fixed step and classify every grid point into two
    disjoint candidate buckets. Never run bisection/false-position once over
    a whole wide domain - scan first, then refine each candidate separately
    (a wide interval can hide an even number of roots, or an asymptote can
    fake a sign change).
      - direct_candidates   : x where |fn(x)| is already <= direct_tol
      - interval_candidates : consecutive (x, x+step) pairs where fn changes sign
    Domain errors mid-scan (e.g. an asymptote) are skipped, not fatal.
    """
    xs = list(np.arange(a, b + step, step))
    vals = [safe_eval(fn, x)[0] for x in xs]

    direct_candidates = [xs[i] for i, v in enumerate(vals)
                          if v is not None and abs(v) <= direct_tol]

    interval_candidates = []
    for i in range(len(xs) - 1):
        v0, v1 = vals[i], vals[i + 1]
        if v0 is None or v1 is None:
            continue
        if v0 * v1 < 0:
            interval_candidates.append((round(xs[i], 10), round(xs[i + 1], 10)))

    return direct_candidates, interval_candidates


def classify_and_verify(fn, candidate, kind, extra_filter=None, residual_tol=1e-6):
    """
    Turn a raw candidate x-value into a reportable verdict row: type
    ('direct' or 'interval'), the residual fn(candidate), and an explicit
    ACCEPT/REJECT decision. Convergence alone is never enough - the residual
    must actually be near zero (guards against bisection/FP "converging" onto
    a sign-flipping singularity, e.g. f(x)=1/x), and any problem-specific
    extra filter (e.g. "accept only if |xr| <= 1e-6") must also pass.
    """
    fx, err = safe_eval(fn, candidate)
    accepted = fx is not None and abs(fx) <= residual_tol
    if accepted and extra_filter is not None:
        accepted = extra_filter(candidate)
    return {'type': kind, 'x': candidate, 'f_x': fx, 'error': err, 'accepted': accepted}


# -----------------------------------------------------------------------------
# BISECTION - Full with table printing + history
# -----------------------------------------------------------------------------

def bisection(xl, xu, tol=0.0001, max_iter=200, verbose=True):
    """
    Bisection method with full iteration table, reading the global f(x).
    Terminates gracefully (returns xr=None) if f is undefined anywhere
    during the run, instead of crashing - e.g. an asymptote inside the
    bracket. Always check the residual f(root) afterwards; convergence of
    the iteration is not the same claim as "this is a genuine root".
    """
    fxl, err = safe_eval(f, xl)
    fxu, err2 = safe_eval(f, xu)
    if fxl is None or fxu is None:
        raise ValueError(f"f is undefined at a bracket endpoint: {err or err2}")
    if fxl * fxu >= 0:
        raise ValueError(
            f"Bisection requires f(xl)*f(xu) < 0.\n"
            f"  f({xl}) = {fxl:.4f},  f({xu}) = {fxu:.4f}"
        )

    xr_old  = None
    history = []

    if verbose:
        print(f"\n{'-'*70}")
        print(f"{'Bisection Method':^70}")
        print(f"{'-'*70}")
        print(f"{'Iter':<6} {'xl':>12} {'xu':>12} {'xr':>12} {'ea (%)':>12} {'f(xr)':>14}")
        print(f"{'-'*70}")

    for i in range(1, max_iter + 1):
        xr = (xl + xu) / 2.0          # <- BISECTION FORMULA: midpoint
        fxr, err = safe_eval(f, xr)
        if fxr is None:
            if verbose:
                print(f"{i:<6} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {'---':>12} {'undefined':>14}")
                print(f"  [TERMINATED] f undefined at xr = {xr:.6f}: {err}")
            return None, history

        # -- Safe approximate relative error calculation ----------------------
        if xr_old is None:
            ea = 100.0
        elif abs(xr) < 1e-12:
            # ea = float('inf')  # Alternative exact-zero infinity sentinel
            ea = abs(xr - xr_old) * 100.0
        else:
            ea = abs((xr - xr_old) / xr) * 100.0

        history.append({'iter': i, 'xl': xl, 'xu': xu,
                        'xr': xr, 'ea': ea, 'fxr': fxr})

        if verbose:
            ea_str = f"{ea:>12.6f}" if xr_old is not None else f"{'---':>12}"
            print(f"{i:<6} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str} {fxr:>14.8f}")

        # -- stopping check ----------------------------------------------------
        if xr_old is not None and ea <= tol:
            break

        # -- update bracket ----------------------------------------------------
        if fxl * fxr < 0:
            xu = xr       # root is in LEFT half  -> shrink upper bound
        else:
            xl, fxl = xr, fxr       # root is in RIGHT half -> shrink lower bound

        xr_old = xr

    if verbose:
        print(f"{'-'*70}")
        print(f"  Converged root approx. {xr:.8f}  (ea = {ea:.6f}%,  iterations = {i})")

    return xr, history


# -----------------------------------------------------------------------------
# FALSE POSITION - Full with table printing + history + stagnation tracking
# -----------------------------------------------------------------------------

def false_position(xl, xu, tol=0.0001, max_iter=500, verbose=True):
    """
    False Position (Regula Falsi) method with full iteration table, reading
    the global f(x). Terminates gracefully (returns xr=None) if f is
    undefined anywhere during the run.
    """
    fl, err = safe_eval(f, xl)
    fu, err2 = safe_eval(f, xu)
    if fl is None or fu is None:
        raise ValueError(f"f is undefined at a bracket endpoint: {err or err2}")
    if fl * fu >= 0:
        raise ValueError(
            f"False Position requires f(xl)*f(xu) < 0.\n"
            f"  f({xl}) = {fl:.4f},  f({xu}) = {fu:.4f}"
        )

    xr_old     = None
    xl_updates = 0
    xu_updates = 0
    history    = []

    if verbose:
        print(f"\n{'-'*70}")
        print(f"{'False Position (Regula Falsi) Method':^70}")
        print(f"{'-'*70}")
        print(f"{'Iter':<6} {'xl':>12} {'xu':>12} {'xr':>12} {'ea (%)':>12} {'f(xr)':>14}")
        print(f"{'-'*70}")

    for i in range(1, max_iter + 1):
        # -- FALSE POSITION FORMULA -------------------------------------------
        xr  = xu - fu * (xl - xu) / (fl - fu)
        fxr, err = safe_eval(f, xr)
        if fxr is None:
            if verbose:
                print(f"{i:<6} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {'---':>12} {'undefined':>14}")
                print(f"  [TERMINATED] f undefined at xr = {xr:.6f}: {err}")
            return None, history, xl_updates, xu_updates

        # -- Safe approximate relative error calculation ----------------------
        if xr_old is None:
            ea = 100.0
        elif abs(xr) < 1e-12:
            # ea = float('inf')  # Alternative exact-zero infinity sentinel
            ea = abs(xr - xr_old) * 100.0
        else:
            ea = abs((xr - xr_old) / xr) * 100.0

        history.append({'iter': i, 'xl': xl, 'xu': xu,
                        'xr': xr, 'ea': ea, 'fxr': fxr})

        if verbose:
            ea_str = f"{ea:>12.6f}" if xr_old is not None else f"{'---':>12}"
            print(f"{i:<6} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str} {fxr:>14.8f}")

        if xr_old is not None and ea <= tol:
            break

        # -- update bracket ----------------------------------------------------
        if fl * fxr < 0:
            xu, fu = xr, fxr;  xu_updates += 1
        else:
            xl, fl = xr, fxr;  xl_updates += 1

        xr_old = xr

    if verbose:
        print(f"{'-'*70}")
        print(f"  Converged root approx. {xr:.8f}  (ea = {ea:.6f}%,  iterations = {i})")
        print(f"  xl moved {xl_updates} times,  xu moved {xu_updates} times")
        if xl_updates == 0 or xu_updates == 0:
            stuck = 'xl' if xl_updates == 0 else 'xu'
            print(f"  [WARNING]  STAGNATION: {stuck} never updated! (expected for concave/convex f)")

    return xr, history, xl_updates, xu_updates


# -----------------------------------------------------------------------------
# FALSE POSITION - Illinois Modification to break stagnation
# -----------------------------------------------------------------------------

def false_position_illinois(xl, xu, tol=0.0001, max_iter=500, verbose=True):
    """
    Modified False Position Method (Illinois variant) to eliminate boundary
    stagnation, reading the global f(x). When one boundary stagnates, we
    halve the function value at that boundary to force the secant line to
    switch sides on the next step.
    """
    fl, err = safe_eval(f, xl)
    fu, err2 = safe_eval(f, xu)
    if fl is None or fu is None:
        raise ValueError(f"f is undefined at a bracket endpoint: {err or err2}")
    if fl * fu >= 0:
        raise ValueError(
            f"False Position Illinois requires f(xl)*f(xu) < 0.\n"
            f"  f({xl}) = {fl:.4f},  f({xu}) = {fu:.4f}"
        )

    xr_old = None
    history = []

    if verbose:
        print(f"\n{'-'*70}")
        print(f"{'False Position (Illinois Method)':^70}")
        print(f"{'-'*70}")
        print(f"{'Iter':<6} {'xl':>12} {'xu':>12} {'xr':>12} {'ea (%)':>12} {'f(xr)':>14}")
        print(f"{'-'*70}")

    for i in range(1, max_iter + 1):
        # False Position Formula with modified fl/fu weight
        xr = xu - (fu * (xl - xu)) / (fl - fu)
        fxr, err = safe_eval(f, xr)
        if fxr is None:
            if verbose:
                print(f"{i:<6} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {'---':>12} {'undefined':>14}")
                print(f"  [TERMINATED] f undefined at xr = {xr:.6f}: {err}")
            return None, history

        # Safe approximate relative error
        if xr_old is None:
            ea = 100.0
        elif abs(xr) < 1e-12:
            # ea = float('inf')  # Alternative exact-zero infinity sentinel
            ea = abs(xr - xr_old) * 100.0
        else:
            ea = abs((xr - xr_old) / xr) * 100.0

        history.append({'iter': i, 'xl': xl, 'xu': xu,
                        'xr': xr, 'ea': ea, 'fxr': fxr})

        if verbose:
            ea_str = f"{ea:>12.6f}" if xr_old is not None else f"{'---':>12}"
            print(f"{i:<6} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str} {fxr:>14.8f}")

        if xr_old is not None and ea <= tol:
            break

        if fl * fxr < 0:
            xu = xr
            fu = fxr
            fl = fl / 2.0  # Halve weight of inactive bound (xl) to shift secant slope
        else:
            xl = xr
            fl = fxr
            fu = fu / 2.0  # Halve weight of inactive bound (xu) to shift secant slope

        xr_old = xr

    if verbose:
        print(f"{'-'*70}")
        print(f"  Converged root (Illinois) ~ {xr:.8f}  (ea = {ea:.6f}%,  iterations = {i})")

    return xr, history


# -----------------------------------------------------------------------------
# MULTI-ROOT: Incremental search + bisection/FP on EACH sub-interval
# -----------------------------------------------------------------------------

def find_sign_change_intervals(fn, a, b, step=0.1):
    """Scan [a,b] and collect all brackets where fn changes sign. Takes fn
    explicitly (unlike the solvers above) so you can scan with a different
    callable than the global f if a question ever needs that."""
    intervals = []
    xs = np.arange(a, b, step)
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i + 1]
        if fn(x1) * fn(x2) < 0:
            intervals.append((round(x1, 10), round(x2, 10)))
    return intervals


def find_all_roots(a, b, step=0.1, method='bisection', tol=0.0001):
    """
    Automatically detect all sign-change brackets in the global f and solve
    each one. method: 'bisection', 'false_position', or 'false_position_illinois'
    Returns a list of roots.
    """
    intervals = find_sign_change_intervals(f, a, b, step)
    print(f"\nFound {len(intervals)} sign-change interval(s): "
          f"{[(round(a,2), round(b,2)) for a, b in intervals]}\n")

    roots = []
    for xl, xu in intervals:
        if method == 'bisection':
            root, _ = bisection(xl, xu, tol=tol, verbose=True)
        elif method == 'false_position_illinois':
            root, _ = false_position_illinois(xl, xu, tol=tol, verbose=True)
        else:
            root, _, _, _ = false_position(xl, xu, tol=tol, verbose=True)
        roots.append(root)

    return roots


# -----------------------------------------------------------------------------
# COMPARISON TABLE: Bisection vs False Position vs Illinois
# -----------------------------------------------------------------------------

def compare_methods(xl, xu, tol=0.0001):
    """Run Bisection, False Position, and Illinois methods (on the global f)
    and print a comparative summary."""
    print(f"\n{'='*85}")
    print(f"{'COMPARISON: Bisection vs False Position vs Illinois':^85}")
    print(f"{'='*85}")

    bi_root, bi_hist               = bisection(xl, xu, tol=tol, verbose=False)
    fp_root, fp_hist, fp_l, fp_u   = false_position(xl, xu, tol=tol, verbose=False)
    fpi_root, fpi_hist             = false_position_illinois(xl, xu, tol=tol, verbose=False)

    bi_ea = bi_hist[-1]['ea']
    fp_ea = fp_hist[-1]['ea']
    fpi_ea = fpi_hist[-1]['ea']

    print(f"\n  {'Method':<20} {'Root':>14} {'Iterations':>12} {'Final f(xr)':>16} {'Final ea (%)':>14}")
    print(f"  {'-'*20} {'-'*14} {'-'*12} {'-'*16} {'-'*14}")
    print(f"  {'Bisection':<20} {bi_root:>14.8f} {len(bi_hist):>12} {f(bi_root):>16.8f} {bi_ea:>14.6f}")
    print(f"  {'False Position':<20} {fp_root:>14.8f} {len(fp_hist):>12} {f(fp_root):>16.8f} {fp_ea:>14.6f}")
    print(f"  {'Illinois FP':<20} {fpi_root:>14.8f} {len(fpi_hist):>12} {f(fpi_root):>16.8f} {fpi_ea:>14.6f}")
    print(f"\n  [Stagnation Info] Standard FP: xl moved {fp_l}x, xu moved {fp_u}x")


# -----------------------------------------------------------------------------
# PLOTTING HELPERS for Root-Finding Methods
# (these take fn explicitly - a numpy-vectorized version of f - since the
#  curve needs to be evaluated over a whole array at once)
# -----------------------------------------------------------------------------

def plot_root_finding(fn, a, b, roots=None, intervals=None,
                      title='Root Finding', fname=None):
    """Combined plot: function curve + optional brackets + root markers."""
    x = np.linspace(a, b, 1000)
    y = fn(x)

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, color='steelblue', linewidth=2, label='f(x)')
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='gray',  linewidth=0.7, linestyle=':')

    if intervals:
        colors = ['orange', 'lightgreen', 'violet']
        for k, (xl, xu) in enumerate(intervals):
            plt.axvspan(xl, xu, color=colors[k % len(colors)], alpha=0.3,
                        label=f'Bracket [{round(xl,2)}, {round(xu,2)}]')

    if roots:
        plt.scatter(roots, [0]*len(roots), color='red', marker='x',
                    s=150, linewidths=2.5, zorder=5, label='Root(s)')

    plt.xlabel('x');  plt.ylabel('f(x)');  plt.title(title)
    plt.legend();     plt.grid(alpha=0.4); plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
    plt.show()


# -----------------------------------------------------------------------------
# == EXAMPLE 1: ln(x) from Previous Year A1/B1/C1 Exam ==
# -----------------------------------------------------------------------------

if __name__ == '__main__':

    print("\n" + "="*70)
    print("EXAMPLE 1: f(x) = ln(x),  interval [1e-4, 1e4],  tol = 0.0001%")
    print("="*70)

    f = lambda x: np.log(x)          # <- redefine the global f for this example

    # Sign verification (required by exam questions)
    print(f"\n  f(1e-4) = {np.log(1e-4):.6f},  f(1e4) = {np.log(1e4):.6f}")
    print(f"  Sign change exists: {np.log(1e-4) * np.log(1e4) < 0}")

    # Run both methods and compare
    compare_methods(1e-4, 1e4, tol=0.0001)

    # Plot on log scale
    x_log = np.logspace(-4, 4, 1000)
    plt.figure(figsize=(8, 5))
    plt.plot(x_log, np.log(x_log), color='royalblue', linewidth=2)
    plt.axhline(0, color='red', linewidth=1.2, linestyle='--', label='y = 0')
    plt.xscale('log')
    plt.xlabel('x (log scale)'); plt.ylabel('f(x) = ln(x)')
    plt.title('f(x) = ln(x) on Logarithmic Scale')
    plt.grid(True, which='both', alpha=0.4); plt.legend()
    plt.tight_layout(); plt.savefig('bisect_fp_ln_logscale.png', dpi=150); plt.show()

    # --------------------------------------------------------------------------
    # EXAMPLE 2: Multi-root function from Section B
    # --------------------------------------------------------------------------

    print("\n" + "="*70)
    print("EXAMPLE 2: f(x) = 0.6*ln(x+1) - sin(1.7x) - 0.08x^2 - 0.08")
    print("           Scan [0, 10] with step 0.1, find ALL roots")
    print("="*70)

    sth = 1.0    # <- CHANGE this to match your exam coefficient
    f  = lambda x: 0.6 * np.log(x + 1) - sth * np.sin(1.7 * x) - 0.08 * x**2 - 0.08

    roots2 = find_all_roots(0, 10, step=0.1, method='bisection', tol=0.0001)
    print(f"\n  ALL ROOTS: {[round(r, 6) for r in roots2 if r is not None]}")

    intervals2 = find_sign_change_intervals(f, 0, 10, step=0.1)
    plot_root_finding(f, 0, 10, roots=roots2, intervals=intervals2,
                      title='Multiple Roots: Section B Function',
                      fname='bisect_fp_section_b.png')

    # --------------------------------------------------------------------------
    # EXAMPLE 3: Chapra Polynomial - auto-scan + false position on FIRST interval
    # --------------------------------------------------------------------------

    print("\n" + "="*70)
    print("EXAMPLE 3: f(x) = 2x^3 - 11.7x^2 + 17.7x - 5")
    print("           Auto-scan, then False Position on FIRST bracket")
    print("="*70)

    f = lambda x: 2*x**3 - 11.7*x**2 + 17.7*x - 5

    intervals3 = find_sign_change_intervals(f, 0, 4, step=0.1)
    print(f"\n  Brackets found: {[(round(a,2), round(b,2)) for a,b in intervals3]}")

    if intervals3:
        xl3, xu3 = intervals3[0]    # use FIRST interval
        root3, hist3, _, _ = false_position(xl3, xu3, tol=0.0001, verbose=True)

    plot_root_finding(f, 0, 4, roots=[root3], intervals=intervals3,
                      title='Chapra Polynomial: Auto-Scan + False Position',
                      fname='bisect_fp_chapra.png')

    # --------------------------------------------------------------------------
    # EXAMPLE 4: Pathological/discontinuous function (exam B2 style) -
    #            direct candidates + interval candidates + acceptance filter
    # --------------------------------------------------------------------------

    print("\n" + "="*70)
    print("EXAMPLE 4: f(x) = (x-2.5)^2*(x+1.052) / (x-3.551)   [-2, 5]")
    print("           coarse_scan + bisection + classify_and_verify")
    print("="*70)

    f = lambda x: ((x - 2.5) ** 2 * (x + 1.052)) / (x - 3.551)

    direct_candidates, interval_candidates = coarse_scan(f, -2, 5, step=0.1, direct_tol=1e-8)
    print(f"\n  Direct candidates: {direct_candidates}")
    print(f"  Interval candidates: {interval_candidates}")

    verdicts = []
    for x0 in direct_candidates:
        verdicts.append(classify_and_verify(f, x0, 'direct',
                                            extra_filter=lambda x: abs(x) <= 1e-6))
    for xl4, xu4 in interval_candidates:
        xr4, _ = bisection(xl4, xu4, tol=0.0001, verbose=False)
        if xr4 is not None:
            verdicts.append(classify_and_verify(f, xr4, 'interval',
                                                extra_filter=lambda x: abs(x) <= 1e-6))

    print(f"\n  {'Type':<10} {'x':>14} {'f(x)':>16} {'Verdict':>10}")
    for v in verdicts:
        fx_str = f"{v['f_x']:.6e}" if v['f_x'] is not None else "undefined"
        verdict_str = "ACCEPT" if v['accepted'] else "REJECT"
        print(f"  {v['type']:<10} {v['x']:>14.6f} {fx_str:>16} {verdict_str:>10}")
