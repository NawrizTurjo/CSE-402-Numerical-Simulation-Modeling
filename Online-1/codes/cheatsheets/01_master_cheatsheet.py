"""
=============================================================================
  MASTER CHEATSHEET - CSE-402 Numerical Methods (Root-Finding + Errors)
=============================================================================

HOW TO USE THIS FILE IN THE EXAM:
  1. Read the question, identify the method(s) and what tables/plots it wants.
  2. Redefine f(x) [and df(x) for NR] at the top for this question.
  3. Copy the relevant function(s) below (or just run this file if it fits).
  4. Print table(s). Save plot(s). State the root/verdict clearly.

WHAT'S IN HERE (see numbered sections):
  (1) f(x)/df(x) template
  (2) Scanning: quick scan, candidate scan (direct root vs interval, for
      pathological/discontinuous functions), find-and-solve-all-roots
  (3) Bisection            - with xl/xu update tracking + undefined-value guard
  (4) False Position       - with xl/xu update tracking + undefined-value guard
  (4b) False Position (Illinois) - fixes stagnation
  (5) Newton-Raphson       - multiplicity m, oscillation trap detection,
      numeric-derivative fallback (no df needed)
  (6) Bairstow's Method    - all roots of a polynomial (NOTE: out of scope for
      this term's slide deck per references/, kept as a bonus)
  (7) Comparison table     - Bisection vs False Position side by side
      (root, iterations, f(xr), xl/xu update counts) - exactly what a
      "compare the two methods" question wants
  (8) Plotting templates   - standard, log-scale, single root+bracket,
      MULTI bracket + multi root (for B1-style "find all roots" questions),
      ea convergence (semilogy), Newton-Raphson tangent-line geometry
  (9) Error & Approximation theory (Lab 1) - true/approx relative error,
      Scarborough tolerance, machine epsilon, Taylor-series truncation demo,
      subtractive cancellation demo, round-off vs truncation tradeoff curve
  (10) Newton-Raphson failure-mode reference card (memorize these!)
  (11) Quick formula reference card

CONVENTION USED THROUGHOUT THIS CODEBASE:
  Solvers read the MODULE-LEVEL f(x) (and df(x) for NR) - they take no f
  argument. Redefine f/df, then just call bisection(xl, xu), etc.
  Plotting helpers take an explicit f_np (vectorized) argument instead,
  since you often want to plot a different view/range than what you solve.
=============================================================================
"""

import math
import cmath
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# (1) DEFINE YOUR FUNCTION HERE - change this for every question!
# =============================================================================

def f(x):
    # EXAMPLES - uncomment the one that matches your exam:
    return x**3 - x - 1                              # x^3 - x - 1 = 0
    # return 0.6*math.log(x+1) - 1.0*math.sin(1.7*x) - 0.08*x**2 - 0.08   # B1-style multi-root
    # return math.pi*x**2*(3*4 - x)/3 - 5             # dipstick: r=4, V=5
    # return 2*x**3 - 11.7*x**2 + 17.7*x - 5          # Chapra polynomial
    # return 1e-12*(math.exp(x/(1.8*0.02585))-1)+x/500 - 0.0002           # diode
    # return ((x-2.5)**2*(x+1.052))/(x-3.551)         # B2-style pathological/asymptote

def df(x):
    # Derivative of f (needed ONLY for Newton-Raphson). Chain rule always.
    return 3*x**2 - 1                                 # derivative of x^3-x-1
    # return 0.6/(x+1) - 1.0*1.7*math.cos(1.7*x) - 0.16*x
    # return math.pi*(8*x - x**2)                     # derivative of dipstick
    # return 6*x**2 - 23.4*x + 17.7
    # return 1e-12/(1.8*0.02585)*math.exp(x/(1.8*0.02585)) + 1/500


# =============================================================================
# (2) SCANNING - always run this before any method!
# =============================================================================

def scan_for_roots(a, b, step=0.1):
    """Scan [a,b] with step size. Returns list of (xl,xu) sign-change brackets."""
    xs = np.arange(a, b, step)
    intervals = []
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i+1]
        if f(x1) * f(x2) < 0:
            intervals.append((round(x1, 6), round(x2, 6)))
    print(f"Sign-change intervals: {intervals}")
    return intervals


def safe_eval(x):
    """Evaluate global f(x), catching domain errors instead of crashing.
    Returns (value, None) on success, (None, error_message) on failure."""
    try:
        return f(x), None
    except (ZeroDivisionError, ValueError, OverflowError) as e:
        return None, str(e)


def coarse_scan(a, b, step=0.1, direct_tol=1e-8):
    """
    For PATHOLOGICAL / discontinuous functions (asymptotes, double roots).
    Classifies grid points into:
      - direct root candidates: |f(x)| already <= direct_tol
      - sign-change interval candidates: (xl, xu) pairs
    Domain errors mid-scan are skipped, not fatal. Never bisect blindly over
    a whole wide domain when f might be undefined somewhere inside it -
    scan first, refine each candidate separately.
    """
    xs = list(np.arange(a, b + step, step))
    vals = [safe_eval(x)[0] for x in xs]
    direct = [xs[i] for i, v in enumerate(vals) if v is not None and abs(v) <= direct_tol]
    intervals = []
    for i in range(len(xs) - 1):
        v0, v1 = vals[i], vals[i+1]
        if v0 is None or v1 is None:
            continue
        if v0 * v1 < 0:
            intervals.append((round(xs[i], 10), round(xs[i+1], 10)))
    return direct, intervals


def classify_and_verify(candidate, kind, extra_filter=None, residual_tol=1e-6):
    """Verdict row for a candidate: type ('direct'/'interval'), residual
    f(candidate), ACCEPT/REJECT. Convergence alone is never enough - the
    residual must be near zero (guards against converging onto a
    sign-flipping singularity/asymptote), and any problem-specific extra
    filter (e.g. exam literally says "accept only if |xr| <= 1e-6") must
    also pass."""
    fx, err = safe_eval(candidate)
    accepted = fx is not None and abs(fx) <= residual_tol
    if accepted and extra_filter is not None:
        accepted = extra_filter(candidate)
    return {'type': kind, 'x': candidate, 'f_x': fx, 'error': err, 'accepted': accepted}


def calc_sig_digit(err):
    """Inverted Scarborough formula: given an actual error, how many
    significant figures does that guarantee? floor(2 - log10(2*err))."""
    if err == 0:
        return 9999
    return math.floor(2 - math.log10(2 * err))


def find_all_roots(a, b, step=0.1, method='bisection', tol=0.0001):
    """
    Scan [a,b] for ALL sign-change brackets, then SOLVE every one of them.
    method: 'bisection', 'false_position', or 'false_position_illinois'
    Returns a list of roots (skips any bracket that terminates on an
    undefined value instead of crashing the whole run).
    """
    intervals = scan_for_roots(a, b, step)
    roots = []
    for xl, xu in intervals:
        if method == 'false_position_illinois':
            root = false_position_illinois(xl, xu, tol=tol)
        elif method == 'false_position':
            result = false_position(xl, xu, tol=tol)
            root = result[0] if result is not None else None
        else:
            result = bisection(xl, xu, tol=tol)
            root = result[0] if result is not None else None
        if root is not None:
            roots.append(root)
    print(f"\nALL ROOTS FOUND ({method}): {[round(r, 6) for r in roots]}")
    return roots


# =============================================================================
# (3) BISECTION METHOD
# =============================================================================

def bisection(xl, xu, tol=0.0001, max_iter=200, verbose=True):
    """
    FORMULA:  xr = (xl + xu) / 2
    STOP:     ea = |xr_new - xr_old| / |xr_new| x 100 <= tol
    UPDATE:   f(xl)*f(xr) < 0 -> xu=xr,  else -> xl=xr
    Tracks xl/xu update counts (needed for method-comparison questions).
    Terminates gracefully (returns None) if f is undefined mid-run.
    """
    if verbose:
        print(f"\n{'-'*68}")
        print(f"{'BISECTION METHOD':^68}")
        print(f"{'-'*68}")
        print(f"{'Iter':<5} {'xl':>12} {'xu':>12} {'xr':>12} {'ea(%)':>12} {'f(xr)':>12}")
        print(f"{'-'*68}")

    xl_updates = 0
    xu_updates = 0
    xr_old = None
    xr = None

    for i in range(1, max_iter + 1):
        xr = (xl + xu) / 2.0          # <- BISECTION FORMULA
        fxr, err = safe_eval(xr)
        if fxr is None:
            if verbose:
                print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {'---':>12} {'undefined':>12}")
                print(f"  [TERMINATED] f undefined at xr = {xr:.6f}: {err}")
            return None

        if xr_old is None:
            ea = 100.0
        elif abs(xr) < 1e-12:
            # ea = float('inf')  # Alternative exact-zero infinity sentinel
            ea = abs(xr - xr_old) * 100.0
        else:
            ea = abs((xr - xr_old) / xr) * 100.0

        if verbose:
            ea_str = f"{ea:.6f}" if xr_old else "---"
            print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str:>12} {fxr:>12.8f}")

        if xr_old and ea <= tol:
            break

        if f(xl) * fxr < 0:
            xu = xr; xu_updates += 1
        else:
            xl = xr; xl_updates += 1
        xr_old = xr

    if verbose:
        print(f"{'-'*68}")
        print(f"  ROOT approx. {xr:.8f}   ea = {ea:.6f}%   Iter = {i}")
        print(f"  xl moved {xl_updates}x,  xu moved {xu_updates}x")

    return xr, i, ea, xl_updates, xu_updates


# =============================================================================
# (4) FALSE POSITION (REGULA FALSI) METHOD
# =============================================================================

def false_position(xl, xu, tol=0.0001, max_iter=500, verbose=True):
    """
    FORMULA:  xr = xu - f(xu)*(xl - xu) / (f(xl) - f(xu))
    STOP:     ea <= tol  (same as bisection)
    Tracks xl/xu update counts - REQUIRED for "compare bisection vs false
    position, how many times did xl/xu update" style exam questions.
    Terminates gracefully (returns None) if f is undefined mid-run.
    """
    if verbose:
        print(f"\n{'-'*68}")
        print(f"{'FALSE POSITION (REGULA FALSI)':^68}")
        print(f"{'-'*68}")
        print(f"{'Iter':<5} {'xl':>12} {'xu':>12} {'xr':>12} {'ea(%)':>12} {'f(xr)':>12}")
        print(f"{'-'*68}")

    xl_updates = 0
    xu_updates = 0
    xr_old = None
    xr = None

    for i in range(1, max_iter + 1):
        fl, fu = f(xl), f(xu)
        xr = xu - fu * (xl - xu) / (fl - fu)     # <- FALSE POSITION FORMULA
        fxr, err = safe_eval(xr)
        if fxr is None:
            if verbose:
                print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {'---':>12} {'undefined':>12}")
                print(f"  [TERMINATED] f undefined at xr = {xr:.6f}: {err}")
            return None

        if xr_old is None:
            ea = 100.0
        elif abs(xr) < 1e-12:
            # ea = float('inf')  # Alternative exact-zero infinity sentinel
            ea = abs(xr - xr_old) * 100.0
        else:
            ea = abs((xr - xr_old) / xr) * 100.0

        if verbose:
            ea_str = f"{ea:.6f}" if xr_old else "---"
            print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str:>12} {fxr:>12.8f}")

        if xr_old and ea <= tol:
            break

        if fl * fxr < 0:
            xu = xr; xu_updates += 1
        else:
            xl = xr; xl_updates += 1
        xr_old = xr

    if verbose:
        print(f"{'-'*68}")
        print(f"  ROOT approx. {xr:.8f}   ea = {ea:.6f}%   Iter = {i}")
        print(f"  xl moved {xl_updates}x,  xu moved {xu_updates}x")
        if xl_updates == 0 or xu_updates == 0:
            stuck = 'xl' if xl_updates == 0 else 'xu'
            print(f"  [WARNING] STAGNATION: {stuck} never updated! (typical for concave/convex f)")

    return xr, i, ea, xl_updates, xu_updates


# =============================================================================
# (4b) FALSE POSITION WITH ILLINOIS MODIFICATION (BREAKS STAGNATION)
# =============================================================================

def false_position_illinois(xl, xu, tol=0.0001, max_iter=500, verbose=True):
    """
    Illinois method to fix stagnation on concave/convex functions.
    If a boundary stagnates, the inactive bound's function value is halved,
    forcing the secant line to switch sides sooner.
    """
    if verbose:
        print(f"\n{'-'*68}")
        print(f"{'FALSE POSITION (ILLINOIS)':^68}")
        print(f"{'-'*68}")
        print(f"{'Iter':<5} {'xl':>12} {'xu':>12} {'xr':>12} {'ea(%)':>12} {'f(xr)':>12}")
        print(f"{'-'*68}")

    fl = f(xl)
    fu = f(xu)
    xr_old = None
    xr = None

    for i in range(1, max_iter + 1):
        xr = xu - fu * (xl - xu) / (fl - fu)
        fxr, err = safe_eval(xr)
        if fxr is None:
            if verbose:
                print(f"  [TERMINATED] f undefined at xr = {xr:.6f}: {err}")
            return None

        if xr_old is None:
            ea = 100.0
        elif abs(xr) < 1e-12:
            ea = abs(xr - xr_old) * 100.0
        else:
            ea = abs((xr - xr_old) / xr) * 100.0

        if verbose:
            ea_str = f"{ea:.6f}" if xr_old else "---"
            print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str:>12} {fxr:>12.8f}")

        if xr_old and ea <= tol:
            break

        if fl * fxr < 0:
            xu = xr; fu = fxr
            fl = fl / 2.0          # halve stagnant bound's weight
        else:
            xl = xr; fl = fxr
            fu = fu / 2.0
        xr_old = xr

    if verbose:
        print(f"{'-'*68}")
        print(f"  ROOT approx. {xr:.8f}   ea = {ea:.6f}%   Iter = {i}")

    return xr


# =============================================================================
# (5) NEWTON-RAPHSON METHOD
# =============================================================================

def _newton_raphson_core(x0, m=1, tol=0.0001, max_iter=100, verbose=True):
    """Shared core: returns (root, history) where history is a list of
    dicts {iter, xi, fxi, dfxi, xi1, ea}. Used by newton_raphson() and by
    the tangent-line plotter so both read the exact same computation."""
    if verbose:
        print(f"\n{'-'*80}")
        print(f"{'NEWTON-RAPHSON METHOD (m=' + str(m) + ')':^80}")
        print(f"{'-'*80}")
        print(f"{'Iter':<5} {'x_i':>12} {'f(x_i)':>14} {'f_prime(x_i)':>14} {'x_i+1':>14} {'ea(%)':>12}")
        print(f"{'-'*80}")

    xi = float(x0)
    seen = [xi]
    history = []
    ea = 100.0

    for i in range(1, max_iter + 1):
        fxi = f(xi)
        dfxi = df(xi)
        if abs(dfxi) < 1e-12:
            if verbose:
                print(f"  [CRITICAL] f'(x) ~ 0 at x={xi:.6f}. Division by zero!")
            break

        xi1 = xi - m * (fxi / dfxi)          # <- NEWTON-RAPHSON FORMULA

        if abs(xi1) < 1e-12:
            # ea = float('inf')  # Alternative exact-zero infinity sentinel
            ea = abs(xi1 - xi) * 100.0
        else:
            ea = abs((xi1 - xi) / xi1) * 100.0

        history.append({'iter': i, 'xi': xi, 'fxi': fxi, 'dfxi': dfxi, 'xi1': xi1, 'ea': ea})
        if verbose:
            print(f"{i:<5} {xi:>12.6f} {fxi:>14.8f} {dfxi:>14.8f} {xi1:>14.8f} {ea:>12.6f}")

        # ping-pong / oscillation trap check
        for past_x in seen[:-1]:
            if abs(xi1 - past_x) < 1e-6:
                if verbose:
                    print(f"\n  [WARNING] Infinite oscillation cycle detected! (jumping back to {xi1:.4f})")
                    print("  Fix: choose a starting guess closer to the true sign-change bracket.")
                return xi1, history

        if ea <= tol:
            xi = xi1
            break
        seen.append(xi1)
        xi = xi1

    if verbose and history:
        print(f"{'-'*80}")
        print(f"  ROOT approx. {xi:.8f}   ea = {ea:.6f}%   Iter = {history[-1]['iter']}")

    return xi, history


def newton_raphson(x0, m=1, tol=0.0001, max_iter=100, verbose=True):
    """
    FORMULA:  x_{i+1} = x_i - m * f(x_i) / f'(x_i)   (m = root multiplicity,
    m=1 for simple roots; use m>1 to restore quadratic speed on a known
    multiple root)
    STOP:     ea <= tol
    Includes cycle-detection trap and safe denominator checks.
    """
    root, _ = _newton_raphson_core(x0, m, tol, max_iter, verbose)
    return root


def newton_raphson_numeric(x0, h=1e-6, tol=0.0001, max_iter=100, verbose=True):
    """
    Newton-Raphson using a NUMERIC derivative (central difference):
      f'(x) ~= (f(x+h) - f(x-h)) / (2h)
    Use this when deriving f'(x) analytically is too painful/error-prone
    under time pressure. Temporarily overrides the global df(x).
    """
    global df
    original_df = df
    df = lambda x: (f(x + h) - f(x - h)) / (2 * h)
    try:
        root, history = _newton_raphson_core(x0, 1, tol, max_iter, verbose)
    finally:
        df = original_df
    return root


# =============================================================================
# (6) BAIRSTOW'S METHOD (Polynomial roots - all at once)
#     NOTE: per references/, out of scope for this term's slide deck.
#     Kept here as a bonus in case a polynomial-roots question appears.
# =============================================================================

def bairstow_all_roots(coeffs, r0=1.0, s0=1.0, tol=0.001, max_iter=100):
    """
    Finds ALL roots of a polynomial (coefficients highest power FIRST).
    e.g. x^4 - 5x^3 + 7x^2 - 5x + 6  ->  coeffs=[1,-5,7,-5,6]

    TABLE FORMAT: iter  r  s  dr  ds  ea(r)%  ea(s)%
    STOP: both ea(r) <= tol AND ea(s) <= tol
    """
    def synth_div(a, r, s):
        n = len(a) - 1
        b = [0.0] * (n + 1)
        b[0] = a[0]
        b[1] = a[1] + r * b[0]
        for i in range(2, n + 1):
            b[i] = a[i] + r * b[i-1] + s * b[i-2]
        return b

    a = [float(c) for c in coeffs]
    roots = []

    while len(a) - 1 >= 3:
        n = len(a) - 1
        r, s = float(r0), float(s0)
        print(f"\n  Finding quadratic factor for degree-{n} polynomial:")
        print(f"  {'iter':<5} {'r':>12} {'s':>12} {'dr':>10} {'ds':>10} {'ea_r%':>10} {'ea_s%':>10}")
        print(f"  {'-'*65}")

        for it in range(1, max_iter + 1):
            b = synth_div(a, r, s)
            b_short = b[:n]
            c = synth_div(b_short, r, s)
            cn2, cn3, cn1 = c[n-2], (c[n-3] if n >= 3 else 0.0), c[n-1]
            bn1, bn = b[n-1], b[n]
            det = cn2 * cn2 - cn3 * cn1
            if abs(det) < 1e-14:
                r += 0.5; s += 0.5; continue
            dr = (-bn1 * cn2 + bn * cn3) / det
            ds = (-bn * cn2 + bn1 * cn1) / det
            r += dr; s += ds
            ea_r = abs(dr / r) * 100 if r != 0 else 1e9
            ea_s = abs(ds / s) * 100 if s != 0 else 1e9
            print(f"  {it:<5} {r:>12.6f} {s:>12.6f} {dr:>10.6f} {ds:>10.6f} {ea_r:>10.4f} {ea_s:>10.4f}")
            if ea_r <= tol and ea_s <= tol:
                break

        disc = r*r + 4*s
        r1 = (r + cmath.sqrt(disc)) / 2
        r2 = (r - cmath.sqrt(disc)) / 2
        roots.extend([r1, r2])
        print(f"\n  -> Roots of x^2-({r:.5f})x-({s:.5f}): {r1},  {r2}")
        b_final = synth_div(a, r, s)
        a = b_final[:n-1]

    if len(a) - 1 == 2:
        a0, a1, a2 = a
        disc = a1*a1 - 4*a0*a2
        sq = cmath.sqrt(disc)
        roots.extend([(-a1+sq)/(2*a0), (-a1-sq)/(2*a0)])
    elif len(a) - 1 == 1:
        roots.append(-a[1] / a[0])

    cleaned = [float(rt.real) if abs(rt.imag) < 1e-9 else rt for rt in roots]
    print(f"\n  ALL ROOTS: {cleaned}")
    return cleaned


def poly_eval(coeffs, x):
    """Evaluate polynomial at x via Horner's method (coeffs highest power first)."""
    val = 0.0 + 0j if isinstance(x, complex) else 0.0
    for c in coeffs:
        val = val * x + c
    return val


def verify_roots(coeffs, roots):
    """Print |f(root)| for each root - should be approx. 0. Sanity check
    after Bairstow (or any solver) before trusting the answer."""
    print(f"\n  {'Root':>25}  {'|f(root)|':>14}")
    print(f"  {'-'*42}")
    for rt in roots:
        fval = poly_eval(coeffs, rt)
        print(f"  {str(rt):>25}  {abs(fval):>14.2e}")


# =============================================================================
# (7) COMPARISON TABLE: Bisection vs False Position
#     Exactly what "compare the two methods: root, iterations, f(xr),
#     xl/xu update counts" questions want (e.g. log-scale ln(x) question).
# =============================================================================

def compare_methods(xl, xu, tol=0.0001):
    """Runs Bisection and False Position (silently) on the same bracket and
    prints one side-by-side comparison table."""
    print(f"\n{'='*90}")
    print(f"{'COMPARISON: Bisection vs False Position':^90}")
    print(f"{'='*90}")

    bi = bisection(xl, xu, tol=tol, verbose=False)
    fp = false_position(xl, xu, tol=tol, verbose=False)

    header = f"{'Method':<18} {'Root':>14} {'Iterations':>12} {'Final f(xr)':>16} {'xl updates':>12} {'xu updates':>12}"
    print(header)
    print("-" * 90)
    if bi is not None:
        bi_root, bi_i, bi_ea, bi_lu, bi_uu = bi
        print(f"{'Bisection':<18} {bi_root:>14.8f} {bi_i:>12} {f(bi_root):>16.8e} {bi_lu:>12} {bi_uu:>12}")
    else:
        print(f"{'Bisection':<18} {'TERMINATED (undefined value hit)':>68}")

    if fp is not None:
        fp_root, fp_i, fp_ea, fp_lu, fp_uu = fp
        print(f"{'False Position':<18} {fp_root:>14.8f} {fp_i:>12} {f(fp_root):>16.8e} {fp_lu:>12} {fp_uu:>12}")
        if fp_lu == 0 or fp_uu == 0:
            print(f"\n  [Stagnation] False Position: {'xl' if fp_lu==0 else 'xu'} never updated "
                  f"-> secant method degraded to one-sided convergence.")
    else:
        print(f"{'False Position':<18} {'TERMINATED (undefined value hit)':>68}")
    print("=" * 90)

    return bi, fp


# =============================================================================
# (8) PLOTTING TEMPLATES - COPY AND ADAPT
# =============================================================================

def plot_standard(f_np, a, b, title='f(x)', fname='graph.png'):
    """Standard plot: function + zero line + grid. Most common exam format."""
    x = np.linspace(a, b, 1000)
    y = f_np(x)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color='steelblue', linewidth=2, label='f(x)')
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel('x'); plt.ylabel('f(x)'); plt.title(title)
    plt.grid(alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.show()


def plot_logscale(f_np, a, b, title='f(x) log scale', fname='graph_log.png'):
    """Log-scale plot: use when x spans many orders of magnitude (e.g. 1e-4 to 1e4)."""
    x = np.logspace(np.log10(a), np.log10(b), 1000)
    y = f_np(x)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color='royalblue', linewidth=2, label='f(x)')
    plt.axhline(0, color='red', linewidth=1, linestyle='--')
    plt.xscale('log')
    plt.xlabel('x (log scale)'); plt.ylabel('f(x)'); plt.title(title)
    plt.grid(True, which='both', alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.show()


def plot_with_root(f_np, a, b, root, xl=None, xu=None, title='Root Found', fname='graph_root.png'):
    """Single-root plot: function + optional bracket shading + root marker."""
    x = np.linspace(a, b, 1000)
    y = f_np(x)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color='steelblue', linewidth=2, label='f(x)')
    plt.axhline(0, color='black', linewidth=1)
    if xl is not None and xu is not None:
        plt.axvspan(xl, xu, color='orange', alpha=0.25, label=f'Bracket [{xl:.2f},{xu:.2f}]')
    plt.scatter([root], [0], color='red', s=120, marker='x',
                linewidths=2.5, zorder=5, label=f'Root approx. {root:.6f}')
    plt.xlabel('x'); plt.ylabel('f(x)'); plt.title(title)
    plt.grid(alpha=0.4); plt.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.show()


def plot_with_brackets(f_np, a, b, intervals=None, roots=None, title='Multi-Root Finding', fname='graph_multi.png'):
    """
    MULTI-root / MULTI-bracket plot - for "find ALL roots in [a,b]"
    questions (B1-style). Shades every candidate interval in a different
    color, scatters every bracket endpoint, and marks every converged root.
    """
    x = np.linspace(a, b, 1000)
    y = f_np(x)
    plt.figure(figsize=(9, 5.5))
    plt.plot(x, y, color='steelblue', linewidth=2, label='f(x)')
    plt.axhline(0, color='black', linewidth=1)

    if intervals:
        colors = ['orange', 'lightgreen', 'violet', 'gold', 'cyan', 'salmon']
        for k, (xl, xu) in enumerate(intervals):
            plt.axvspan(xl, xu, color=colors[k % len(colors)], alpha=0.3,
                        label=f'Bracket [{round(xl,2)}, {round(xu,2)}]')
            plt.scatter([xl, xu], [f_np(np.array([xl, xu]))], color='red', s=40, zorder=5)

    if roots:
        plt.scatter(roots, [0]*len(roots), color='red', marker='x',
                    s=150, linewidths=2.5, zorder=6, label='Root(s)')

    plt.xlabel('x'); plt.ylabel('f(x)'); plt.title(title)
    plt.legend(fontsize=8); plt.grid(alpha=0.4); plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.show()


def plot_convergence(errors, title='Convergence of ea', fname='convergence.png'):
    """
    Semilog plot of ea vs iteration - "show convergence" questions.
    Straight-line decay on log-y -> linear convergence (bisection/FP).
    Rapidly steepening near the end -> quadratic convergence (Newton-Raphson).
    """
    iters = list(range(1, len(errors) + 1))
    plt.figure(figsize=(7, 4))
    plt.semilogy(iters, errors, 'o-', color='darkred', markersize=5)
    plt.xlabel('Iteration number'); plt.ylabel('ea (%) [log scale]')
    plt.title(title); plt.grid(alpha=0.4, which='both'); plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.show()


def plot_newton_raphson(x0, a, b, m=1, tol=0.0001, title='Newton-Raphson Tangent Geometry', fname='nr_tangents.png'):
    """
    Visualizes NR's tangent-line geometry: draws the tangent at each x_i,
    showing where it crosses the x-axis to land on x_{i+1}.
    """
    root, history = _newton_raphson_core(x0, m, tol, max_iter=100, verbose=False)

    x_range = np.linspace(a, b, 600)
    f_np = np.vectorize(f)
    y_range = f_np(x_range)

    plt.figure(figsize=(9, 5))
    plt.plot(x_range, y_range, color='steelblue', linewidth=2, label='f(x)')
    plt.axhline(0, color='black', linewidth=1)

    colors_t = plt.cm.Reds(np.linspace(0.4, 0.9, max(len(history), 1)))
    for k, h in enumerate(history[:6]):     # only show first 6 tangents
        xi, fxi, dfxi = h['xi'], h['fxi'], h['dfxi']
        x_tan = np.array([min(a, xi - 0.5), max(b, xi + 0.5)])
        y_tan = dfxi * (x_tan - xi) + fxi
        plt.plot(x_tan, y_tan, '--', color=colors_t[k], alpha=0.7, linewidth=1.2)
        plt.scatter([xi], [fxi], color=colors_t[k], s=50, zorder=5)

    plt.scatter([root], [0], color='red', marker='x', s=150, linewidths=2.5,
                zorder=6, label=f'Root approx. {root:.6f}')
    plt.scatter([x0], [f(x0)], color='green', s=80, zorder=5, label=f'x0 = {x0}')

    plt.xlim(a, b)
    plt.xlabel('x'); plt.ylabel('f(x)'); plt.title(title)
    plt.legend(); plt.grid(alpha=0.4); plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.show()


# =============================================================================
# (9) ERROR & APPROXIMATION THEORY (Lab 1 - errors/approximations slide deck)
# =============================================================================

def true_error(true_val, approx_val):
    """E_t = true - approx. Sign matters (positive = over-estimated)."""
    return true_val - approx_val


def true_relative_error_pct(true_val, approx_val):
    """epsilon_t% = |true-approx|/|true| x 100. Needs the real answer
    (not available in practice, but used to grade a demo/example)."""
    return abs(true_val - approx_val) / abs(true_val) * 100.0


def approx_relative_error_pct(x_new, x_old):
    """
    ea% = |x_new-x_old|/|x_new| x 100. THE practical stopping criterion -
    used when the true value is unknown, comparing consecutive iterates.
    """
    if x_new == 0:
        return float('inf')
    return abs((x_new - x_old) / x_new) * 100.0


def scarborough_tolerance(n_sig_figs):
    """
    Scarborough criterion: the tolerance (%) that guarantees at least
    n_sig_figs correct significant figures.
    Formula: es = 0.5 x 10^(2-n)  (%).  e.g. n=3 -> es=0.05%, n=4 -> es=0.005%
    """
    return 0.5 * 10 ** (2 - n_sig_figs)


def compute_machine_epsilon():
    """Keep halving until (1 + eps/2) == 1 in floating-point. The
    fundamental precision limit of a 64-bit float (~2.22e-16)."""
    eps = 1.0
    while (1.0 + eps / 2.0) != 1.0:
        eps /= 2.0
    return eps


def taylor_exp(x, n_terms):
    """Approximates e^x with n_terms of the Taylor/Maclaurin series:
    e^x = 1 + x + x^2/2! + x^3/3! + ...  Truncation error shrinks as
    n_terms grows."""
    total = 0.0
    term = 1.0
    for k in range(n_terms):
        total += term
        term *= x / (k + 1)
    return total


def demo_taylor_truncation(x=1.0, max_terms=10):
    """Print a table of Taylor-series truncation error for e^x shrinking
    as n_terms grows. True value used: math.exp(x)."""
    true_val = math.exp(x)
    print(f"\nTrue value e^{x} = {true_val:.10f}")
    print(f"  {'n_terms':>8}  {'approx':>14}  {'E_t':>14}  {'eps_t (%)':>12}")
    for n in range(1, max_terms + 1):
        approx = taylor_exp(x, n)
        et = true_error(true_val, approx)
        eps_t = true_relative_error_pct(true_val, approx)
        print(f"  {n:>8}  {approx:>14.8f}  {et:>14.8f}  {eps_t:>12.6f}")


def demo_subtractive_cancellation():
    """Classic round-off hazard: sqrt(x+1)-sqrt(x) loses significant digits
    for large x. Rearranged stable form: 1/(sqrt(x+1)+sqrt(x))."""
    print("\nSubtractive cancellation:  sqrt(x+1) - sqrt(x)  vs stable form")
    print(f"  {'x':>12}  {'naive':>18}  {'stable':>18}  {'diff':>12}")
    for x in [1e0, 1e4, 1e8, 1e12]:
        naive = math.sqrt(x + 1) - math.sqrt(x)
        stable = 1.0 / (math.sqrt(x + 1) + math.sqrt(x))
        print(f"  {x:>12.0e}  {naive:>18.12f}  {stable:>18.12f}  {abs(naive-stable):>12.2e}")


def demo_error_tradeoff(x_target=1.0, fname='error_tradeoff.png'):
    """
    Round-off vs truncation error tradeoff for forward-difference numerical
    differentiation of e^x. Small h -> truncation shrinks but round-off
    (subtractive cancellation, then dividing by tiny h) grows. Finds and
    plots the optimal step size h at the bottom of the classic V-curve.
    """
    true_deriv = math.exp(x_target)
    h_values = np.logspace(0, -20, 100)
    errors = []
    for h in h_values:
        approx_deriv = (math.exp(x_target + h) - math.exp(x_target)) / h
        errors.append(abs(true_deriv - approx_deriv))

    min_idx = int(np.argmin(errors))
    optimal_h = h_values[min_idx]
    optimal_error = errors[min_idx]
    print(f"\nOptimal step size h = {optimal_h:.2e},  minimum error = {optimal_error:.4e}")

    plt.figure(figsize=(8, 5))
    plt.loglog(h_values, errors, color='darkblue', linewidth=2.5, label='Total Numerical Error')
    plt.axvline(optimal_h, color='green', linestyle=':', label=f'Optimal h ~ {optimal_h:.1e}')
    plt.xlabel('Step Size h (log scale)'); plt.ylabel('Absolute Error (log scale)')
    plt.title('Truncation vs Round-off Error Tradeoff Curve')
    plt.gca().invert_xaxis()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend(loc='lower left'); plt.tight_layout()
    plt.savefig(fname, dpi=150); plt.show()
    return optimal_h, optimal_error


# =============================================================================
# (10) NEWTON-RAPHSON FAILURE MODES - MUST MEMORIZE (examiners test these!)
# =============================================================================

def nr_failure_modes_reference():
    print("""
    +---------------------------------------------------------------------+
    |  NEWTON-RAPHSON FAILURE MODES                                       |
    +---------------------------------------------------------------------+
    |  Trap        Function        x0    Symptom              Fix         |
    |  2-Cycle     x^3-2x+2        0     Infinite 0->1->0->1  x0=-1.5     |
    |  Zero deriv  sin(x)          pi/2  Division by zero     Shift x0    |
    |  Diverge     cbrt(x)         any   Doubles each step    Use bisect  |
    |  Overshoot   arctan(x)       1.5   Shoots to +-inf      Plot first  |
    |  Slow        (x-2)^3         any   Linear not quadratic Use m>1     |
    +---------------------------------------------------------------------+
    | 1. f'(x_i) = 0          -> division by zero (horizontal tangent)    |
    | 2. Oscillation 0->1->0->1 -> 2-cycle trap                           |
    | 3. f(x) = cbrt(x) near 0 -> diverges, x_(n+1) = -2*x_n              |
    | 4. Flat-tailed functions -> shoots to infinity                      |
    | 5. Multiple roots        -> loses quadratic speed, degrades linear  |
    +---------------------------------------------------------------------+
    """)


# =============================================================================
# (11) QUICK REFERENCE - ERROR & METHOD FORMULAS (print in exam if asked)
# =============================================================================

def error_reference():
    print("""
    +-----------------------------------------------------------------+
    |               ERROR FORMULA REFERENCE CARD                      |
    +-----------------------------------------------------------------+
    |  True Error:         E_t  = true - approx                       |
    |  True Rel. Error:    epsilon_t% = |true-approx|/|true| x 100    |
    |  Approx Rel. Error:  ea% = |x_new-x_old|/|x_new| x 100          |
    |  Scarborough (n SF): es% = 0.5 x 10^(2-n) %                     |
    |                                                                 |
    |  Bisection formula:      xr = (xl + xu) / 2                     |
    |  False Position formula: xr = xu - f(xu)*(xl-xu)/(f(xl)-f(xu))  |
    |  Newton-Raphson formula: x_{i+1} = x_i - m*f(x_i)/f'(x_i)       |
    |  Bairstow corrections:   D = c[n-2]^2 - c[n-3]*c[n-1]           |
    |                          dr = (-b[n-1]*c[n-2]+b[n]*c[n-3])/D    |
    |                          ds = (-b[n]*c[n-2]+b[n-1]*c[n-1])/D    |
    +-----------------------------------------------------------------+
    """)


# =============================================================================
# MAIN - light-weight demo of the key pieces. Comment sections out freely.
# =============================================================================

if __name__ == '__main__':
    error_reference()
    nr_failure_modes_reference()

    # -- STEP 1: Always plot the function first ---------------------------------
    f_np = lambda x: x**3 - x - 1
    plot_standard(f_np, a=0, b=3, title='f(x) = x^3 - x - 1', fname='master_graph.png')

    # -- STEP 2: Find intervals, solve, compare ----------------------------------
    intervals = scan_for_roots(0, 3, step=0.1)
    if intervals:
        xl, xu = intervals[0]
        bi, fp = compare_methods(xl, xu, tol=0.0001)
        if bi is not None:
            plot_with_root(f_np, 0, 3, root=bi[0], xl=xl, xu=xu,
                            title='Bisection Root', fname='master_bisect.png')

        # Newton-Raphson:
        root_nr = newton_raphson(x0=1.5, tol=0.0001)
        plot_newton_raphson(x0=1.5, a=0, b=3, title='NR Tangent Geometry', fname='master_nr_tangents.png')

    # -- STEP 3: Multi-root example (B1-style) -----------------------------------
    sth = 1.0
    f_multiroot_np = lambda x: 0.6*np.log(x+1) - sth*np.sin(1.7*x) - 0.08*x**2 - 0.08
    def f_multiroot(x):
        return 0.6*math.log(x+1) - sth*math.sin(1.7*x) - 0.08*x**2 - 0.08
    _original_f = f                       # save before swapping
    globals()['f'] = f_multiroot          # swap global f for this example
    multi_intervals = scan_for_roots(0, 10, step=0.1)
    multi_roots = find_all_roots(0, 10, step=0.1, method='bisection', tol=0.0001)
    plot_with_brackets(f_multiroot_np, 0, 10, intervals=multi_intervals, roots=multi_roots,
                        title='Section B: Multi-Root Scan', fname='master_multiroot.png')
    globals()['f'] = _original_f          # restore

    # -- STEP 4: Error/approximation theory demos --------------------------------
    print(f"\nMachine epsilon: {compute_machine_epsilon():.3e}")
    demo_taylor_truncation()
    demo_subtractive_cancellation()
    demo_error_tradeoff()

    # -- STEP 5: Bairstow (polynomial only, bonus/out-of-scope) ------------------
    coeffs = [1, -6, 11, -6]   # x^3 - 6x^2 + 11x - 6
    roots_poly = bairstow_all_roots(coeffs, r0=1.0, s0=1.0, tol=0.001)
    verify_roots(coeffs, roots_poly)
