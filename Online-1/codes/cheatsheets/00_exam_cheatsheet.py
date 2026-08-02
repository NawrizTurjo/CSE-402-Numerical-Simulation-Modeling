"""
=============================================================================
  EXAM-DAY CHEATSHEET - CSE-402 Online Lab Assignment 1
=============================================================================

HOW TO USE THIS FILE IN THE EXAM:
  1. Read the question and identify which method to use.
  2. Find the function template below and copy it.
  3. Replace f(x), df(x), xl, xu, x0 with the exam values.
  4. Run. Print table. Done.

COMMON TRICK: If the exam says "plot the graph first", always do that.
Teachers mark on graph quality! Use the plot templates at the bottom.
=============================================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# (1) DEFINE YOUR FUNCTION HERE - change this for every question!
# =============================================================================

def f(x):
    # EXAMPLES - uncomment the one that matches your exam:
    return x**3 - x - 1                              # x^3 - x - 1 = 0
    # return 0.6*math.log(x+1) - 1.0*math.sin(1.7*x) - 0.08*x**2 - 0.08
    # return math.pi*x**2*(3*4 - x)/3 - 5            # dipstick: r=4, V=5
    # return 2*x**3 - 11.7*x**2 + 17.7*x - 5         # Chapra polynomial
    # return 1e-12*(math.exp(x/(1.8*0.02585))-1)+x/500 - 0.0002  # diode

def df(x):
    # Derivative of f (needed ONLY for Newton-Raphson)
    return 3*x**2 - 1                                 # derivative of x^3-x-1
    # return 0.6/(x+1) - 1.0*1.7*math.cos(1.7*x) - 0.16*x
    # return math.pi*(8*x - x**2)                     # derivative of dipstick
    # return 6*x**2 - 23.4*x + 17.7
    # return 1e-12/(1.8*0.02585)*math.exp(x/(1.8*0.02585)) + 1/500


# =============================================================================
# (2) QUICK SCAN - find sign-change intervals (always run this first!)
# =============================================================================

def scan_for_roots(a, b, step=0.1):
    """Scan [a,b] with step size. Print all brackets where f changes sign."""
    xs = np.arange(a, b, step)
    intervals = []
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i+1]
        if f(x1) * f(x2) < 0:
            intervals.append((round(x1, 6), round(x2, 6)))
    print(f"Sign-change intervals: {intervals}")
    return intervals


# =============================================================================
# (2)b CANDIDATE SCANNING & ACCEPTANCE FILTERING
#      (needed for pathological/discontinuous-function questions, e.g. B2)
# =============================================================================

def safe_eval(x):
    """Evaluate global f(x), catching domain errors (div-by-zero, log of a
    non-positive number, overflow) instead of crashing. Returns (value, None)
    on success, (None, error_message) on failure."""
    try:
        return f(x), None
    except (ZeroDivisionError, ValueError, OverflowError) as e:
        return None, str(e)

def coarse_scan(a, b, step=0.1, direct_tol=1e-8):
    """
    Scan [a,b] and classify grid points into direct root candidates
    (|f(x)| already <= direct_tol) and sign-change interval candidates.
    Never bisect once over a whole wide domain - scan first, refine each
    candidate separately. Domain errors mid-scan are skipped, not fatal.
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
    sign-flipping singularity), and any problem-specific extra filter
    (e.g. |xr| <= 1e-6) must also pass."""
    fx, err = safe_eval(candidate)
    accepted = fx is not None and abs(fx) <= residual_tol
    if accepted and extra_filter is not None:
        accepted = extra_filter(candidate)
    return {'type': kind, 'x': candidate, 'f_x': fx, 'error': err, 'accepted': accepted}


# =============================================================================
# (3) BISECTION METHOD
# =============================================================================

def bisection(xl, xu, tol=0.0001, max_iter=200):
    """
    FORMULA:  xr = (xl + xu) / 2
    STOP:     ea = |xr_new - xr_old| / |xr_new| x 100 <= tol
    UPDATE:   f(xl)*f(xr) < 0 -> xu=xr,  else -> xl=xr
    Terminates gracefully (returns None) if f is undefined mid-run.
    """
    print(f"\n{'-'*68}")
    print(f"{'BISECTION METHOD':^68}")
    print(f"{'-'*68}")
    print(f"{'Iter':<5} {'xl':>12} {'xu':>12} {'xr':>12} {'ea(%)':>12} {'f(xr)':>12}")
    print(f"{'-'*68}")

    xr_old = None
    for i in range(1, max_iter+1):
        xr  = (xl + xu) / 2.0          # <- BISECTION FORMULA
        fxr, err = safe_eval(xr)
        if fxr is None:
            print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {'---':>12} {'undefined':>12}")
            print(f"  [TERMINATED] f undefined at xr = {xr:.6f}: {err}")
            return None

        # Safe error calculation near zero
        if xr_old is None:
            ea = 100.0
        elif abs(xr) < 1e-12:
            # ea = float('inf')  # Alternative exact-zero infinity sentinel
            ea = abs(xr - xr_old) * 100.0
        else:
            ea = abs((xr - xr_old)/xr)*100.0

        ea_str = f"{ea:.6f}" if xr_old else "---"
        print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str:>12} {fxr:>12.8f}")
        if xr_old and ea <= tol: break
        if f(xl)*fxr < 0:
            xu = xr
        else:
            xl = xr
        xr_old = xr

    print(f"{'-'*68}")
    print(f"  ROOT approx. {xr:.8f}   ea = {ea:.6f}%   Iter = {i}")
    return xr


# =============================================================================
# (4) FALSE POSITION (REGULA FALSI) METHOD
# =============================================================================

def false_position(xl, xu, tol=0.0001, max_iter=500):
    """
    FORMULA:  xr = xu - f(xu)*(xl - xu) / (f(xl) - f(xu))
    STOP:     ea <= tol  (same as bisection)
    """
    print(f"\n{'-'*68}")
    print(f"{'FALSE POSITION (REGULA FALSI)':^68}")
    print(f"{'-'*68}")
    print(f"{'Iter':<5} {'xl':>12} {'xu':>12} {'xr':>12} {'ea(%)':>12} {'f(xr)':>12}")
    print(f"{'-'*68}")

    xr_old = None
    for i in range(1, max_iter+1):
        fl, fu = f(xl), f(xu)
        xr  = xu - fu*(xl-xu)/(fl-fu)  # <- FALSE POSITION FORMULA
        fxr, err = safe_eval(xr)
        if fxr is None:
            print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {'---':>12} {'undefined':>12}")
            print(f"  [TERMINATED] f undefined at xr = {xr:.6f}: {err}")
            return None

        # Safe error calculation near zero
        if xr_old is None:
            ea = 100.0
        elif abs(xr) < 1e-12:
            # ea = float('inf')  # Alternative exact-zero infinity sentinel
            ea = abs(xr - xr_old) * 100.0
        else:
            ea = abs((xr - xr_old)/xr)*100.0

        ea_str = f"{ea:.6f}" if xr_old else "---"
        print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str:>12} {fxr:>12.8f}")
        if xr_old and ea <= tol: break
        if fl*fxr < 0:
            xu = xr
        else:
            xl = xr
        xr_old = xr

    print(f"{'-'*68}")
    print(f"  ROOT approx. {xr:.8f}   ea = {ea:.6f}%   Iter = {i}")
    return xr


# =============================================================================
# (4)b FALSE POSITION WITH ILLINOIS MODIFICATION (BREAKS STAGNATION)
# =============================================================================

def false_position_illinois(xl, xu, tol=0.0001, max_iter=500):
    """
    Illinois method to solve stagnation problem on concave/convex functions.
    If boundary stagnates, inactive bound function value is halved.
    """
    print(f"\n{'-'*68}")
    print(f"{'FALSE POSITION (ILLINOIS)':^68}")
    print(f"{'-'*68}")
    print(f"{'Iter':<5} {'xl':>12} {'xu':>12} {'xr':>12} {'ea(%)':>12} {'f(xr)':>12}")
    print(f"{'-'*68}")

    fl = f(xl)
    fu = f(xu)
    xr_old = None
    
    for i in range(1, max_iter+1):
        xr  = xu - fu*(xl-xu)/(fl-fu)
        fxr, err = safe_eval(xr)
        if fxr is None:
            print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {'---':>12} {'undefined':>12}")
            print(f"  [TERMINATED] f undefined at xr = {xr:.6f}: {err}")
            return None

        if xr_old is None:
            ea = 100.0
        elif abs(xr) < 1e-12:
            # ea = float('inf')  # Alternative exact-zero infinity sentinel
            ea = abs(xr - xr_old) * 100.0
        else:
            ea = abs((xr - xr_old)/xr)*100.0
            
        ea_str = f"{ea:.6f}" if xr_old else "---"
        print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str:>12} {fxr:>12.8f}")
        if xr_old and ea <= tol: break
        
        if fl * fxr < 0:
            xu = xr; fu = fxr
            fl = fl / 2.0  # Halve the weight of stagnant bound xl
        else:
            xl = xr; fl = fxr
            fu = fu / 2.0  # Halve the weight of stagnant bound xu
        xr_old = xr

    print(f"{'-'*68}")
    print(f"  ROOT approx. {xr:.8f}   ea = {ea:.6f}%   Iter = {i}")
    return xr


# =============================================================================
# (5) NEWTON-RAPHSON METHOD (WITH MULTIPLICITY m & LOOP DETECTION)
# =============================================================================

def newton_raphson(x0, m=1, tol=0.0001, max_iter=100):
    """
    FORMULA:  x_{i+1} = x_i - m * f(x_i) / f'(x_i)
    STOP:     ea <= tol
    WARNING:  Includes cycle detection trap and safe denominator checks.
    """
    print(f"\n{'-'*80}")
    print(f"{'NEWTON-RAPHSON METHOD (m=' + str(m) + ')':^80}")
    print(f"{'-'*80}")
    print(f"{'Iter':<5} {'x_i':>12} {'f(x_i)':>14} {'f_prime(x_i)':>14} {'x_i+1':>14} {'ea(%)':>12}")
    print(f"{'-'*80}")

    xi = float(x0)
    history = [xi]
    ea = 100.0
    
    for i in range(1, max_iter+1):
        fxi  = f(xi)
        dfxi = df(xi)
        if abs(dfxi) < 1e-12:
            print(f"  [CRITICAL] f'(x) ~ 0 at x={xi:.6f}. Division by zero!"); break
            
        xi1 = xi - m * (fxi/dfxi)            # <- NEWTON-RAPHSON FORMULA
        
        # Safe approximate relative error
        if abs(xi1) < 1e-12:
            # ea = float('inf')  # Alternative exact-zero infinity sentinel
            ea = abs(xi1 - xi) * 100.0
        else:
            ea = abs((xi1-xi)/xi1)*100.0
            
        print(f"{i:<5} {xi:>12.6f} {fxi:>14.8f} {dfxi:>14.8f} {xi1:>14.8f} {ea:>12.6f}")
        
        # Ping-pong / oscillation check
        for past_x in history[:-1]:
            if abs(xi1 - past_x) < 1e-6:
                print(f"\n  [WARNING] [WARNING] Infinite oscillation cycle detected! (Jumping back to {xi1:.4f})")
                print("  Fix: Choose a starting guess closer to the true sign-change bracket.")
                return xi1
                
        if ea <= tol: xi = xi1; break
        history.append(xi1)
        xi = xi1

    print(f"{'-'*80}")
    print(f"  ROOT approx. {xi:.8f}   ea = {ea:.6f}%   Iter = {i}")
    return xi


# =============================================================================
# (6) BAIRSTOW'S METHOD (Polynomial roots - all at once)
# =============================================================================

def bairstow_all_roots(coeffs, r0=1.0, s0=1.0, tol=0.001, max_iter=100):
    """
    Finds ALL roots of a polynomial (coefficients highest power FIRST).
    e.g. x^4 - 5x^3 + 7x^2 - 5x + 6  ->  coeffs=[1,-5,7,-5,6]

    TABLE FORMAT: iter  r  s  dr  ds  ea(r)%  ea(s)%
    STOP: both ea(r) <= tol AND ea(s) <= tol
    """
    import cmath

    def synth_div(a, r, s):
        n = len(a) - 1
        b = [0.0] * (n+1)
        b[0] = a[0]
        b[1] = a[1] + r*b[0]
        for i in range(2, n+1):
            b[i] = a[i] + r*b[i-1] + s*b[i-2]
        return b

    a     = [float(c) for c in coeffs]
    roots = []

    while len(a) - 1 >= 3:
        n = len(a) - 1
        r, s = float(r0), float(s0)
        print(f"\n  Finding quadratic factor for degree-{n} polynomial:")
        print(f"  {'iter':<5} {'r':>12} {'s':>12} {'dr':>10} {'ds':>10} {'ea_r%':>10} {'ea_s%':>10}")
        print(f"  {'-'*65}")

        for it in range(1, max_iter+1):
            b = synth_div(a, r, s)
            b_short = b[:n]
            c = synth_div(b_short, r, s)
            cn2, cn3, cn1 = c[n-2], (c[n-3] if n>=3 else 0.0), c[n-1]
            bn1, bn = b[n-1], b[n]
            det = cn2*cn2 - cn3*cn1
            if abs(det) < 1e-14: r+=0.5; s+=0.5; continue
            dr = (-bn1*cn2 + bn*cn3) / det
            ds = (-bn*cn2  + bn1*cn1) / det
            r += dr;  s += ds
            ea_r = abs(dr/r)*100 if r!=0 else 1e9
            ea_s = abs(ds/s)*100 if s!=0 else 1e9
            print(f"  {it:<5} {r:>12.6f} {s:>12.6f} {dr:>10.6f} {ds:>10.6f} {ea_r:>10.4f} {ea_s:>10.4f}")
            if ea_r <= tol and ea_s <= tol: break

        disc = r*r + 4*s
        r1 = (r + cmath.sqrt(disc))/2;  r2 = (r - cmath.sqrt(disc))/2
        roots.extend([r1, r2])
        print(f"\n  -> Roots of x^2-({r:.5f})x-({s:.5f}): {r1},  {r2}")
        b_final = synth_div(a, r, s)
        a = b_final[:n-1]

    if len(a)-1 == 2:
        a0,a1,a2 = a; disc = a1*a1-4*a0*a2
        sq = cmath.sqrt(disc)
        roots.extend([(-a1+sq)/(2*a0), (-a1-sq)/(2*a0)])
    elif len(a)-1 == 1:
        roots.append(-a[1]/a[0])

    # clean up tiny imaginary parts
    cleaned = [float(rt.real) if abs(rt.imag)<1e-9 else rt for rt in roots]
    print(f"\n  ALL ROOTS: {cleaned}")
    return cleaned


# =============================================================================
# (7) PLOTTING TEMPLATES - COPY AND ADAPT
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
    """Log-scale plot: use when x spans many orders of magnitude."""
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
    """Plot function, optional bracket shading, and mark the root."""
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


# =============================================================================
# (8) QUICK REFERENCE - ERROR FORMULAS (print in exam if asked)
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
    |  Newton-Raphson formula: x_{i+1} = x_i - f(x_i)/f'(x_i)         |
    +-----------------------------------------------------------------+
    """)


# =============================================================================
# MAIN - uncomment the method you need in the exam
# =============================================================================

if __name__ == '__main__':
    error_reference()

    # -- STEP 1: Always plot the function first ---------------------------------
    f_np = lambda x: x**3 - x - 1   # numpy version (vectorized)
    plot_standard(f_np, a=0, b=3, title='f(x) = x^3 - x - 1', fname='exam_graph.png')

    # -- STEP 2: Find intervals -------------------------------------------------
    intervals = scan_for_roots(0, 3, step=0.1)

    # -- STEP 3: Solve with the required method ---------------------------------
    if intervals:
        xl, xu = intervals[0]

        # Bisection:
        root_bi = bisection(xl, xu, tol=0.0001)
        plot_with_root(f_np, 0, 3, root=root_bi, xl=xl, xu=xu,
                       title='Bisection Root', fname='exam_bisect.png')

        # False Position:
        root_fp = false_position(xl, xu, tol=0.0001)

        # Newton-Raphson (needs x0 and df):
        root_nr = newton_raphson(x0=1.5, tol=0.0001)

    # Bairstow (polynomial only):
    # coeffs = [1, -6, 11, -6]   # x^3 - 6x^2 + 11x - 6
    # bairstow_all_roots(coeffs, r0=1.0, s0=1.0, tol=0.001)
