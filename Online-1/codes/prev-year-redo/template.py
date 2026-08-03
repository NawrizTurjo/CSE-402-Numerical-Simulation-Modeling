import os
import math
import numpy as np
import matplotlib.pyplot as plt



def f(x):
    # x^3 - 6x^2 + 11x - 6.2
    return np.power(x, 3) - 6 * np.power(x, 2) + 11 * x - 6.2
    # return x-3

def df(x):
    # 3x^2 - 12x + 11
    return 3 * np.power(x, 2) - 12 * x + 11

# def df(x,h=0.0001):
#     return (f(x+h)-f(x))/h

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

# intervals = scan_for_roots(0, 5, step=0.1)

def safe_eval(x, func=f):
    """Evaluate global f(x), catching domain errors (div-by-zero, log of a
    non-positive number, overflow) instead of crashing. Returns (value, None)
    on success, (None, error_message) on failure."""
    try:
        return func(x), None
    except (ZeroDivisionError, ValueError, OverflowError) as e:
        return None, str(e)

def coarse_scan(a, b, step=0.1, direct_tol=1e-8):
    """
    Scan [a,b] and classify grid points into direct root candidates
    (|f(x)| already <= direct_tol) and sign-change interval candidates.
    Never bisect once over a whole wide domain - scan first, refine each
    candidate separately. Domain errors mid-scan are skipped, not fatal.
    direct->they are directly candidate
    interval->the root lies in the interval
    """
    xs = list(np.arange(a, b + step, step))

    # vals = [safe_eval(x)[0] for x in xs]
    vals = []
    for x in xs:
        val, err = safe_eval(x)
        vals.append(val)
    # print(f"vals: {vals}")
    
    # direct = [xs[i] for i, v in enumerate(vals) if v is not None and abs(v) <= direct_tol]
    direct = []
    for i,v in enumerate(vals):
        if v is not None and abs(v)<=direct_tol:
            direct.append(xs[i])
            # print(f"threshold_val: {v}")
    
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


# direct, intervals = coarse_scan(0, 5, step=0.1)
# print(f"direct: {direct}")
# print(f"intervals: {intervals}")

# extra = lambda x: x > 0
# candidate = 1.121114934
# result = classify_and_verify(candidate, "interval", extra_filter=extra)
# print(result)

# direct, intervals = coarse_scan(0,5,0.1)

# for left, right in intervals:
#     candidate = (left+right)/2
#     result = classify_and_verify(candidate,"interval")
#     print(result)

# roots = [
#     1.121114934,
#     1.790851152,
#     3.088033915
# ]
# for r in roots:
#     print(classify_and_verify(r, "interval"))


def calc_sig_digit(err):
    """Inverted Scarborough formula: given an actual error, how many
    significant figures does that guarantee? floor(2 - log10(2*err))."""
    if err == 0:
        return 9999
    return math.floor(2 - math.log10(2 * np.abs(err)))

# err = 0.05
# print(calc_sig_digit(err))

def find_all_roots(a, b, step=0.1, method='bisection', tol=0.0001, m=1, getInterval = False):
    """
    Scan [a,b] for ALL sign-change brackets, then SOLVE every one of them.
    method: 'bisection', 'false_position', or 'false_position_illinois'
    Returns a list of roots (skips any bracket that terminates on an
    undefined value instead of crashing the whole run).
    """
    direct, intervals = coarse_scan(a,b,step)
    roots = []
    for r in direct:
        roots.append(r)
    for xl, xu in intervals:
        if method == 'false_position_illinois':
            result = false_position_illinois(xl, xu, tol=tol)
            if result is not None:
                root = result[0]
            else:
                root = None
        elif method == 'false_position':
            result = false_position(xl, xu, tol=tol)
            # root = result[0] if result is not None else None
            if result is not None:
                root = result[0]
            else:
                root = None
        elif method == 'false_position_opt':
            result = false_position_opt(xl, xu, tol=tol)
            # root = result[0] if result is not None else None
            if result is not None:
                root = result[0]
            else:
                root = None
        elif method == 'newton_raphson':
            x0 = (xl + xu) / 2.0
            root = newton_raphson(x0, m, tol=tol)
        elif method == 'newton_raphson_numeric':
            x0 = (xl + xu) / 2.0
            root = newton_raphson_numeric(x0, m=m, tol=tol)
        else:
            result = bisection(xl, xu, tol=tol)
            # root = result[0] if result is not None else None
            if result is not None:
                root = result[0]
            else:
                root = None
        if root is not None:
            roots.append(root)
            
    print(f"\nDirect roots: {direct}")
    print(f"Intervals: {intervals}")
    
    print(f"\nALL ROOTS FOUND ({method}): {[round(r, 6) for r in roots]}")

    if getInterval:
        return roots, intervals
    return roots

    
def calculate_m_sig_tol(m):
    return 0.5 * np.power(10.0,2-m)

# tol = calculate_m_sig_tol(6)
# print(tol)

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
    history = []

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

        history.append({'iter': i, 'xl': xl, 'xu': xu, 'xr': xr, 'fxr': fxr, 'ea': ea})

        if verbose:
            # Table Print
            ea_str = f"{ea:.6f}" if xr_old is not None else "---"
            print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str:>12} {fxr:>12.8f}")

        # Stopping Criteria
        if xr_old is not None and ea <= tol:
            break

        # safe version for xl
        fxl, err = safe_eval(xl)
        if fxl is None:
            if verbose:
                print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {'---':>12} {'undefined':>12}")
                print(f"  [TERMINATED] f undefined at xr = {xr:.6f}: {err}")
            return None

        if fxl * fxr < 0:
            xu = xr; xu_updates += 1
        else:
            xl = xr; xl_updates += 1
        xr_old = xr

    if verbose:
        print(f"{'-'*68}")
        print(f"  ROOT approx. {xr:.8f}   ea = {ea:.6f}%   Iter = {i}")
        print(f"  xl moved {xl_updates}x,  xu moved {xu_updates}x")

    return xr, i, ea, xl_updates, xu_updates, history

# roots = find_all_roots(a=0,b=5,step=0.1,method='bisection',tol=calculate_m_sig_tol(6))
# print(roots)
# print(calculate_m_sig_tol(4))
# print(1e-8)
# for root in roots:
#     print(classify_and_verify(candidate=root,kind='interval',residual_tol=calculate_m_sig_tol(6)))
# for root in roots:
#     print(classify_and_verify(candidate=root,kind='interval',residual_tol=1e-6))

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
    history = []

    for i in range(1, max_iter + 1):
        fl, fu = f(xl), f(xu)
        xr = xu - fu * (xu - xl) / (fu - fl)     # <- FALSE POSITION FORMULA
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

        history.append({'iter': i, 'xl': xl, 'xu': xu, 'xr': xr, 'fxr': fxr, 'ea': ea})

        if verbose:
            ea_str = f"{ea:.6f}" if xr_old is not None else "---"
            print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str:>12} {fxr:>12.8f}")

        if xr_old is not None and ea <= tol:
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

    return xr, i, ea, xl_updates, xu_updates, history

# roots = find_all_roots(a=0,b=5,step=0.1,method='false_position',tol=calculate_m_sig_tol(6))
# print(roots)
# for root in roots:
#     print(classify_and_verify(candidate=root,kind='interval',residual_tol=calculate_m_sig_tol(6)))

def false_position_illinois(xl, xu, tol=0.0001, max_iter=500, verbose=True):
    """
    Illinois method to fix stagnation on concave/convex functions.
    If a boundary stagnates, the inactive bound's function value is halved,
    forcing the secant line to switch sides sooner.
    Tracks xl/xu update counts - REQUIRED for "compare bisection vs false
    position, how many times did xl/xu update" style exam questions.
    Terminates gracefully (returns None) if f is undefined mid-run.
    """
    if verbose:
        print(f"\n{'-'*68}")
        print(f"{'FALSE POSITION (ILLINOIS)':^68}")
        print(f"{'-'*68}")
        print(f"{'Iter':<5} {'xl':>12} {'xu':>12} {'xr':>12} {'ea(%)':>12} {'f(xr)':>12}")
        print(f"{'-'*68}")

    xl_updates = 0
    xu_updates = 0
    history = []

    # calculate once, saved, otherwise halving er effect haray jabe (safe initial eval)
    fl, err_l = safe_eval(xl)
    fu, err_u = safe_eval(xu)
    if fl is None or fu is None:
        if verbose:
            print(f"  [TERMINATED] Initial bracket undefined: xl_err={err_l}, xu_err={err_u}")
        return None
    
    xr_old = None
    xr = None

    for i in range(1, max_iter + 1):
        denom = fl - fu
        if abs(denom) < 1e-15:
            if verbose:
                print(f"  [TERMINATED] Division by zero: fl == fu at Iteration {i}")
            return None

        xr = xu - fu * (xl - xu) / denom
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

        history.append({'iter': i, 'xl': xl, 'xu': xu, 'xr': xr, 'fxr': fxr, 'ea': ea})

        if verbose:
            ea_str = f"{ea:.6f}" if xr_old is not None else "---"
            print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str:>12} {fxr:>12.8f}")

        if xr_old is not None and ea <= tol:
            break

        if fl * fxr < 0:
            xu = xr; fu = fxr
            fl = fl / 2.0          # halve stagnant bound's weight
            xu_updates += 1
        else:
            xl = xr; fl = fxr
            fu = fu / 2.0
            xl_updates += 1
        xr_old = xr

    if verbose:
        print(f"{'-'*68}")
        print(f"  ROOT approx. {xr:.8f}   ea = {ea:.6f}%   Iter = {i}")
        print(f"  xl moved {xl_updates}x,  xu moved {xu_updates}x")
        # if xl_updates == 0 or xu_updates == 0:
        #     stuck = 'xl' if xl_updates == 0 else 'xu'
        #     print(f"  [WARNING] STAGNATION: {stuck} never updated! (typical for concave/convex f)")

    return xr, i, ea, xl_updates, xu_updates, history

def false_position_opt(xl, xu, tol=0.0001, max_iter=500, verbose=True):
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
    history = []

    # calculate once (safe initial eval)
    fl, err_l = safe_eval(xl)
    fu, err_u = safe_eval(xu)
    if fl is None or fu is None:
        if verbose:
            print(f"  [TERMINATED] Initial bracket undefined: xl_err={err_l}, xu_err={err_u}")
        return None

    for i in range(1, max_iter + 1):
        denom = fu - fl
        if abs(denom) < 1e-15:
            if verbose:
                print(f"  [TERMINATED] Division by zero: fu == fl at Iteration {i}")
            return None

        xr = xu - fu * (xu - xl) / denom     # <- FALSE POSITION FORMULA
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

        history.append({'iter': i, 'xl': xl, 'xu': xu, 'xr': xr, 'fxr': fxr, 'ea': ea})

        if verbose:
            ea_str = f"{ea:.6f}" if xr_old is not None else "---"
            print(f"{i:<5} {xl:>12.6f} {xu:>12.6f} {xr:>12.6f} {ea_str:>12} {fxr:>12.8f}")

        if xr_old is not None and ea <= tol:
            break

        if fl * fxr < 0:
            xu = xr; xu_updates += 1
            fu = fxr
        else:
            xl = xr; xl_updates += 1
            fl = fxr
        xr_old = xr

    if verbose:
        print(f"{'-'*68}")
        print(f"  ROOT approx. {xr:.8f}   ea = {ea:.6f}%   Iter = {i}")
        print(f"  xl moved {xl_updates}x,  xu moved {xu_updates}x")
        if xl_updates == 0 or xu_updates == 0:
            stuck = 'xl' if xl_updates == 0 else 'xu'
            print(f"  [WARNING] STAGNATION: {stuck} never updated! (typical for concave/convex f)")

    return xr, i, ea, xl_updates, xu_updates, history

# roots_fp = find_all_roots(a=0,b=5,step=0.1,method='false_position',tol=calculate_m_sig_tol(6))
# roots_opt = find_all_roots(a=0,b=5,step=0.1,method='false_position_opt',tol=calculate_m_sig_tol(6))
# roots_ill = find_all_roots(a=0,b=5,step=0.1,method='false_position_illinois',tol=calculate_m_sig_tol(6))
    
# # print(roots_fp)
# for root in roots_fp:
#     print(classify_and_verify(candidate=root,kind='interval',residual_tol=calculate_m_sig_tol(6)))
# # print(roots_opt)
# for root in roots_opt:
#     print(classify_and_verify(candidate=root,kind='interval',residual_tol=calculate_m_sig_tol(6)))
# # print(roots_ill)
# for root in roots_ill:
#     print(classify_and_verify(candidate=root,kind='interval',residual_tol=calculate_m_sig_tol(6)))

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
        fxi, err_f = safe_eval(xi, func=f)
        dfxi, err_df = safe_eval(xi, func=df)
        if fxi is None:
            if verbose:
                print(f"  [TERMINATED] Cannot evaluate f(xi) at x={xi:.6f}: {err_f}")
            return None, history
        
        if dfxi is None:
            if verbose:
                print(f"  [TERMINATED] Cannot evaluate df(xi) at x={xi:.6f}: {err_df}")
            return None, history

        if abs(dfxi) < 1e-12:
            if verbose:
                print(f"  [CRITICAL] f'(x) ~ 0 at x={xi:.6f}. Division by zero!")
            # break
            return None, history

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
            # if seen before then oscillation hocche
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


def newton_raphson_numeric(x0, m=1, h=1e-6, tol=0.0001, max_iter=100, verbose=True):
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
        root, history = _newton_raphson_core(x0=x0, m=m, tol=tol, max_iter=max_iter, verbose=verbose)
    finally:
        df = original_df
    return root

# roots_nr = find_all_roots(a=0,b=5,step=0.1,method='newton_raphson',tol=calculate_m_sig_tol(6))
# roots_nrn = find_all_roots(a=0,b=5,step=0.1,method='newton_raphson_numeric',tol=calculate_m_sig_tol(6))
    
# # print(roots_nr)
# for root in roots_nr:
#     print(classify_and_verify(candidate=root,kind='newton_raphson',residual_tol=calculate_m_sig_tol(6)))
# for root in roots_nrn:
#     print(classify_and_verify(candidate=root,kind='newton_raphson_numeric',residual_tol=calculate_m_sig_tol(6)))


def compare_methods(xl, xu, tol=0.0001, m=1):
    """Runs Bisection, False Position, and Newton-Raphson (silently) on the same bracket and
    prints one side-by-side comparison table."""
    print(f"\n{'='*90}")
    print(f"{'COMPARISON: Bisection vs False Position vs Newton-Raphson':^90}")
    print(f"{'='*90}")

    bi = bisection(xl, xu, tol=tol, verbose=False)
    fp = false_position_opt(xl, xu, tol=tol, verbose=False)
    x0 = (xl + xu) / 2.0
    nr = _newton_raphson_core(x0, m=m, tol=tol, verbose=False)

    header = f"{'Method':<18} {'Root':>14} {'Iterations':>12} {'Final f(xr)':>16} {'xl updates':>12} {'xu updates':>12}"
    print(header)
    print("-" * 90)
    if bi is not None:
        bi_root, bi_i, bi_ea, bi_lu, bi_uu, bi_hist = bi
        fbi, _ = safe_eval(bi_root)
        fbi_str = f"{fbi:>16.8e}" if fbi is not None else f"{'undefined':>16}"
        print(f"{'Bisection':<18} {bi_root:>14.8f} {bi_i:>12} {fbi_str} {bi_lu:>12} {bi_uu:>12}")
    else:
        print(f"{'Bisection':<18} {'TERMINATED (undefined value hit)':>68}")

    if fp is not None:
        fp_root, fp_i, fp_ea, fp_lu, fp_uu, fp_hist = fp
        ffp, _ = safe_eval(fp_root)
        ffp_str = f"{ffp:>16.8e}" if ffp is not None else f"{'undefined':>16}"
        print(f"{'False Position':<18} {fp_root:>14.8f} {fp_i:>12} {ffp_str} {fp_lu:>12} {fp_uu:>12}")
        if fp_lu == 0 or fp_uu == 0:
            print(f"\n  [Stagnation] False Position: {'xl' if fp_lu==0 else 'xu'} never updated "
                  f"-> secant method degraded to one-sided convergence.")
    else:
        print(f"{'False Position':<18} {'TERMINATED (undefined value hit)':>68}")

    if nr is not None:
        nr_root, nr_history = nr
        if nr_root is not None and nr_history:
            nr_i = len(nr_history)
            fnr, _ = safe_eval(nr_root)
            fnr_str = f"{fnr:>16.8e}" if fnr is not None else f"{'undefined':>16}"
            print(f"{'Newton-Raphson':<18} {nr_root:>14.8f} {nr_i:>12} {fnr_str} {'---':>12} {'---':>12}")
        else:
            print(f"{'Newton-Raphson':<18} {'TERMINATED (undefined value / zero derivative)':>68}")
    else:
        print(f"{'Newton-Raphson':<18} {'TERMINATED (undefined value / zero derivative)':>68}")
    print("=" * 90)

    return bi, fp, nr
    
# _, intervals = coarse_scan(0,5,0.1)
# for xl, xu in intervals:
#     compare_methods(xl,xu)

def _save_figure(fname, dpi=150):
    """Ensures figures are saved inside the fig/ directory."""
    os.makedirs('fig', exist_ok=True)
    if not (fname.startswith('fig/') or fname.startswith('fig\\') or os.path.dirname(fname)):
        fname = os.path.join('fig', fname)
    plt.savefig(fname, dpi=dpi)

def plot_standard(f_np, a, b, title='f(x)', fname='fig/graph.png'):
    """Standard plot: function + zero line + grid. Most common exam format."""
    x = np.linspace(a, b, 1000)
    y = f_np(x)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color='steelblue', linewidth=2, label='f(x)')
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel('x'); plt.ylabel('f(x)'); plt.title(title)
    plt.grid(alpha=0.4); plt.legend(); plt.tight_layout()
    _save_figure(fname, dpi=150); plt.show()

# plot_standard(
#     np.vectorize(f),
#     a=0,
#     b=5,
#     title="Function Plot",
#     fname="01_standard.png"
# )

def plot_logscale(f_np, a, b, title='f(x) log scale', fname='fig/graph_log.png'):
    """Log-scale plot: use when x spans many orders of magnitude (e.g. 1e-4 to 1e4)."""
    # x = np.logspace(np.log10(a), np.log10(b), 1000)
    x = np.linspace(a,b,1000)
    y = f_np(x)
    plt.figure(figsize=(8, 5))
    # plt.plot(x, y, color='royalblue', linewidth=2, label='f(x)')
    plt.semilogx(x, y, color='royalblue', linewidth=2, label='f(x)')
    plt.axhline(0, color='red', linewidth=1, linestyle='--')
    plt.xscale('log')
    plt.xlabel('x (log scale)'); plt.ylabel('f(x)'); plt.title(title)
    plt.grid(True, which='both', alpha=0.4); plt.legend(); plt.tight_layout()
    _save_figure(fname, dpi=150); plt.show()


# plot_logscale(
#     np.vectorize(f),
#     a=0.1,
#     b=10,
#     title="Function (Log Scale)",
#     fname="02_log.png"
# )

def plot_with_root(f_np, a, b, root, xl=None, xu=None, title='Root Found', fname='fig/graph_root.png'):
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
    _save_figure(fname, dpi=150); plt.show()

# _, intervals = coarse_scan(0,5,0.1)
# xl, xu = intervals[0]
# bi = bisection(xl=xl, xu=xu, verbose=False)
# root = bi[0]

# plot_with_root(
#     np.vectorize(f),
#     a=0,
#     b=5,
#     root=root,
#     xl=1.1,
#     xu=1.2,
#     title="Bisection Root",
#     fname="03_root.png"
# )


def plot_with_brackets(f_np, a, b, intervals=None, roots=None, title='Multi-Root Finding', fname='fig/graph_multi.png'):
    """
    MULTI-root / MULTI-bracket plot - for "find ALL roots in [a,b]"
    questions (B1-style). Shades every candidate interval using a continuous
    matplotlib colormap (same style as plot_newton_raphson).
    """
    x = np.linspace(a, b, 1000)
    y = f_np(x)
    plt.figure(figsize=(9, 5.5))
    plt.plot(x, y, color='steelblue', linewidth=2, label='f(x)')
    plt.axhline(0, color='black', linewidth=1)

    if intervals:
        colors_b = plt.cm.Oranges(np.linspace(0.3, 0.8, max(len(intervals), 1)))
        for k, (xl, xu) in enumerate(intervals):
            plt.axvspan(xl, xu, color=colors_b[k], alpha=0.35,
                        label=f'Bracket [{round(xl, 2)}, {round(xu, 2)}]')
            plt.scatter([xl, xu], [f_np(xl), f_np(xu)], color=colors_b[k], s=40, zorder=5)

    if roots:
        plt.scatter(roots, [0]*len(roots), color='red', marker='x',
                    s=150, linewidths=2.5, zorder=6, label='Root(s)')

    plt.xlabel('x'); plt.ylabel('f(x)'); plt.title(title)
    plt.legend(fontsize=8); plt.grid(alpha=0.4); plt.tight_layout()
    _save_figure(fname, dpi=150); plt.show()

# def plot_with_brackets(f_np, a, b, intervals=None, roots=None, title='Multi-Root Finding', fname='fig/graph_multi.png'):
#     x = np.linspace(a, b, 1000)
#     plt.figure(figsize=(9, 5.5))
#     plt.plot(x, f_np(x), color='steelblue', linewidth=2, label='f(x)')
#     plt.axhline(0, color='black', linewidth=1)

#     # Simple 2-line bracket shading (no enumerate / no % colors)
#     if intervals:
#         for xl, xu in intervals:
#             plt.axvspan(xl, xu, color='orange', alpha=0.3, label=f'Bracket [{xl:.1f},{xu:.1f}]')
#             plt.scatter([xl, xu], [f_np(xl), f_np(xu)], color='red', s=40)

#     # Mark converged roots
#     if roots:
#         plt.scatter(roots, [0]*len(roots), color='red', marker='x', s=150, linewidths=2.5, label='Root(s)')

#     plt.xlabel('x'); plt.ylabel('f(x)'); plt.title(title)
#     plt.grid(alpha=0.4); plt.legend(fontsize=8); plt.tight_layout()
#     _save_figure(fname, dpi=150); plt.show()


# direct, intervals = coarse_scan(0,5,0.1)

# roots = find_all_roots(
#     0,
#     5,
#     step=0.1,
#     method='false_position_illinois',
#     tol=calculate_m_sig_tol(6)
# )

# plot_with_brackets(
#     np.vectorize(f),
#     0,
#     5,
#     intervals=intervals,
#     roots=roots,
#     title="All Roots",
#     fname="04_all_roots.png"
# )


def plot_convergence(errors, title='Convergence of ea', fname='fig/convergence.png'):
    """
    Semilog plot of ea vs iteration - "show convergence" questions.
    Accepts either a list of numerical errors or a history list (dict items with 'ea' key).
    Straight-line decay on log-y -> linear convergence (bisection/FP).
    Rapidly steepening near the end -> quadratic convergence (Newton-Raphson).
    """
    if errors and isinstance(errors[0], dict):
        errors = [h['ea'] for h in errors if 'ea' in h]
        # new_errors = []
        # for h in errors:
        #     if 'ea' in h:
        #         new_errors.append(h['ea'])
        # errors=new_errors

    iters = list(range(1, len(errors) + 1))
    plt.figure(figsize=(7, 4))
    plt.semilogy(iters, errors, 'o-', color='darkred', markersize=5)
    plt.xlabel('Iteration number'); plt.ylabel('ea (%) [log scale]')
    plt.title(title); plt.grid(alpha=0.4, which='both'); plt.tight_layout()
    _save_figure(fname, dpi=150); plt.show()

# bi = bisection(1.1,1.2,verbose=False)
# # nr = _newton_raphson_core(x0=1.15,verbose=False)

# plot_convergence(
#     bi[-1],
#     # nr[-1],
#     title="Bisection Convergence",
#     fname="05_bisection_conv.png"
# )

# fp = false_position_opt(1.1,1.2,verbose=False)

# plot_convergence(
#     fp[-1],
#     title="False Position Convergence",
#     fname="06_false_position_conv.png"
# )


# ill = false_position_illinois(1.1,1.2,verbose=False)

# plot_convergence(
#     ill[-1],
#     title="Illinois Convergence",
#     fname="07_illinois_conv.png"
# )

# root, hist = _newton_raphson_core(
#     x0=1.15,
#     verbose=False
# )

# plot_convergence(
#     hist,
#     title="Newton-Raphson Convergence",
#     fname="08_nr_conv.png"
# )


def plot_newton_raphson(x0, a, b, m=1, tol=0.0001, title='Newton-Raphson Tangent Geometry', fname='fig/nr_tangents.png'):
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
    for k, h in enumerate(history[:]):     # only show first 6 tangents
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
    _save_figure(fname, dpi=150); plt.show()

# plot_newton_raphson(
#     x0=1.15,
#     a=0,
#     b=5,
#     title="Newton-Raphson Geometry",
#     fname="09_nr_tangent.png"
# )
    
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

# nr_failure_modes_reference()

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

# error_reference()

def true_error(true_val, approx_val):
    """E_t = true - approx. Sign matters (positive = over-estimated)."""
    return true_val - approx_val

# Et = true_error(3.14159, 3.14)


def true_relative_error_pct(true_val, approx_val):
    """epsilon_t% = |true-approx|/|true| x 100. Needs the real answer
    (not available in practice, but used to grade a demo/example)."""
    return abs(true_val - approx_val) / abs(true_val) * 100.0

# eps_t = true_relative_error_pct(3.14159, 3.14)


def approx_relative_error_pct(x_new, x_old):
    """
    ea% = |x_new-x_old|/|x_new| x 100. THE practical stopping criterion -
    used when the true value is unknown, comparing consecutive iterates.
    """
    if x_new == 0:
        return float('inf')
    return abs((x_new - x_old) / x_new) * 100.0

# ea = approx_relative_error_pct(1.324, 1.318)


def scarborough_tolerance(n_sig_figs):
    """
    Scarborough criterion: the tolerance (%) that guarantees at least
    n_sig_figs correct significant figures.
    Formula: es = 0.5 x 10^(2-n)  (%).  e.g. n=3 -> es=0.05%, n=4 -> es=0.005%
    """
    return 0.5 * 10 ** (2 - n_sig_figs)

# es = scarborough_tolerance(n_sig_figs=4)


def compute_machine_epsilon():
    """Keep halving until (1 + eps/2) == 1 in floating-point. The
    fundamental precision limit of a 64-bit float (~2.22e-16)."""
    eps = 1.0
    while (1.0 + eps / 2.0) != 1.0:
        eps /= 2.0
    return eps

# eps_mach = compute_machine_epsilon()


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

# approx = taylor_exp(x=1.0, n_terms=5)


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

# demo_taylor_truncation()

def demo_subtractive_cancellation():
    """Classic round-off hazard: sqrt(x+1)-sqrt(x) loses significant digits
    for large x. Rearranged stable form: 1/(sqrt(x+1)+sqrt(x))."""
    print("\nSubtractive cancellation:  sqrt(x+1) - sqrt(x)  vs stable form")
    print(f"  {'x':>12}  {'naive':>18}  {'stable':>18}  {'diff':>12}")
    for x in [1e0, 1e4, 1e8, 1e12]:
        naive = math.sqrt(x + 1) - math.sqrt(x)
        stable = 1.0 / (math.sqrt(x + 1) + math.sqrt(x))
        print(f"  {x:>12.0e}  {naive:>18.12f}  {stable:>18.12f}  {abs(naive-stable):>12.2e}")

# demo_subtractive_cancellation()

def demo_error_tradeoff(x_target=1.0, fname='fig/error_tradeoff.png'):
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
    _save_figure(fname, dpi=150); plt.show()
    return optimal_h, optimal_error

# opt_h, opt_err = demo_error_tradeoff()


def riemann_sum(f_func, a, b, n):
    """
    Left-endpoint Riemann sum approximation for integral of f(x) over [a, b].
    Demonstrates truncation error shrinking as N (number of rectangles) increases.
    """
    x_pts = np.linspace(a, b, n, endpoint=False)
    width = (b - a) / n
    f_np = np.vectorize(f_func)
    return np.sum(f_np(x_pts) * width)

# area = riemann_sum(math.sin, a=0, b=math.pi, n=50)


def forward_diff(f_func, x, h):
    """Forward difference: f'(x) ~= (f(x+h) - f(x)) / h. Order O(h)."""
    return (f_func(x + h) - f_func(x)) / h

# df_fwd = forward_diff(math.sin, x=1.0, h=0.01)


def backward_diff(f_func, x, h):
    """Backward difference: f'(x) ~= (f(x) - f(x-h)) / h. Order O(h)."""
    return (f_func(x) - f_func(x - h)) / h

# df_bwd = backward_diff(math.sin, x=1.0, h=0.01)


def central_diff(f_func, x, h):
    """Central difference: f'(x) ~= (f(x+h) - f(x-h)) / (2h). Order O(h^2)."""
    return (f_func(x + h) - f_func(x - h)) / (2.0 * h)

# df_cnt = central_diff(math.sin, x=1.0, h=0.01)


def diff_truncation_table(f_func, fprime_exact, x0, h_list=[1.0, 0.1, 0.01, 0.001, 0.0001]):
    """
    Prints a comparison table of Forward, Backward, and Central differences
    against the true derivative f'(x0) across a list of step sizes h.
    """
    exact = fprime_exact(x0)
    print(f"\n{'='*75}")
    print(f"NUMERICAL DIFFERENTIATION TRUNCATION ERROR (x = {x0}, exact f'(x) = {exact:.10f})")
    print(f"{'='*75}")
    print(f"{'h':<10}{'Forward':>15}{'Backward':>15}{'Central':>15}{'Forward Err':>15}")
    print("-" * 75)
    for h in h_list:
        fwd = forward_diff(f_func, x0, h)
        bwd = backward_diff(f_func, x0, h)
        cnt = central_diff(f_func, x0, h)
        err = abs(exact - fwd)
        print(f"{h:<10.5f}{fwd:>15.8f}{bwd:>15.8f}{cnt:>15.8f}{err:>15.8e}")
    print("=" * 75)

# diff_truncation_table(math.sin, math.cos, x0=1.0)


def maclaurin_series(term_recurrence, x, first_term=1.0, sig_digits=3, max_terms=100):
    """
    Generalized Maclaurin / Taylor series solver with Scarborough stopping.
    term_recurrence(prev_term, n, x) -> returns the n-th term from the (n-1)-th.
    Example for e^x: maclaurin_series(lambda prev, n, x: prev * x / n, x=1.2, sig_digits=3)
    """
    term = first_term
    S = term
    es = scarborough_tolerance(sig_digits)
    history = [{'term': 0, 'S': S, 'ea': 100.0, 'sig': 0}]

    for n in range(1, max_terms + 1):
        term = term_recurrence(term, n, x)
        S_new = S + term
        ea = abs((S_new - S) / S_new) * 100.0 if S_new != 0 else 0.0
        sig = calc_sig_digit(ea)
        history.append({'term': n, 'S': S_new, 'ea': ea, 'sig': sig})
        S = S_new
        if ea <= es:
            break
    return S, history

# S, history = maclaurin_series(lambda prev, n, x: prev * x / n, x=1.2, sig_digits=3)


def print_history_table(history, title=None):
    """
    Prints a formatted iteration table from solver history dict list.
    Auto-detects solver type: Bisection/False Position, Newton-Raphson, or Maclaurin.
    """
    if not history:
        print("No history data to display.")
        return

    sample = history[0]

    # Case 1: Newton-Raphson history
    if 'xi' in sample or 'xi1' in sample:
        t_name = title if title else "NEWTON-RAPHSON ITERATION TABLE"
        print(f"\n{'-'*84}")
        print(f"{t_name:^84}")
        print(f"{'-'*84}")
        print(f"{'Iter':<6}{'x_i':>14}{'f(x_i)':>18}{'f_prime(x_i)':>18}{'x_i+1':>16}{'ea (%)':>12}")
        print(f"{'-'*84}")
        for h in history:
            ea_str = "---" if h['iter'] == 1 else f"{h['ea']:.6f}"
            print(f"{h['iter']:<6}{h['xi']:>14.6f}{h['fxi']:>18.6e}{h['dfxi']:>18.6e}{h['xi1']:>16.6f}{ea_str:>12}")
        print(f"{'-'*84}")

    # Case 2: Bisection / False Position history
    elif 'xl' in sample and 'xr' in sample:
        t_name = title if title else "BRACKETING METHOD ITERATION TABLE"
        print(f"\n{'-'*76}")
        print(f"{t_name:^76}")
        print(f"{'-'*76}")
        print(f"{'Iter':<6}{'xl':>14}{'xu':>14}{'xr':>14}{'ea (%)':>14}{'f(xr)':>14}")
        print(f"{'-'*76}")
        for h in history:
            ea_str = "---" if h['iter'] == 1 else f"{h['ea']:.6f}"
            print(f"{h['iter']:<6}{h['xl']:>14.6f}{h['xu']:>14.6f}{h['xr']:>14.6f}{ea_str:>14}{h['fxr']:>14.4e}")
        print(f"{'-'*76}")

    # Case 3: Maclaurin / Series history
    elif 'term' in sample or 'S' in sample:
        t_name = title if title else "SERIES EXPANSION TABLE"
        print(f"\n{'-'*70}")
        print(f"{t_name:^70}")
        print(f"{'-'*70}")
        print(f"{'Term (k)':<10}{'Partial Sum S_k':>22}{'ea (%)':>18}{'Sig Digits':>20}")
        print(f"{'-'*70}")
        for h in history:
            ea_str = "---" if h['term'] == 0 else f"{h['ea']:.6f}"
            sig_str = "---" if h['term'] == 0 else f"{h['sig']} digits"
            print(f"{h['term']:<10}{h['S']:>22.10f}{ea_str:>18}{sig_str:>20}")
        print(f"{'-'*70}")

# print_history_table(history, title="Iteration Table")

# demo_error_tradeoff()


# error_reference()
# nr_failure_modes_reference()

# -- STEP 1: Always plot the function first ---------------------------------
# f_np = lambda x: x**3 - x - 1
# plot_standard(f_np, a=0, b=3, title='f(x) = x^3 - x - 1', fname='master_graph.png')

# -- STEP 2: Find intervals, solve, compare ----------------------------------
# intervals = scan_for_roots(0, 3, step=0.1)
# if intervals:
#     xl, xu = intervals[0]
#     bi, fp, nr = compare_methods(xl, xu, tol=0.0001)
    
#     if bi is not None:
#         plot_with_root(f_np, 0, 3, root=bi[0], xl=xl, xu=xu,
#                         title='Bisection Root', fname='master_bisect.png')
#         plot_convergence(bi[5], title='Bisection Convergence', fname='master_bisect_conv.png')

#     if fp is not None:
#         plot_with_root(f_np, 0, 3, root=fp[0], xl=xl, xu=xu,
#                         title='False Position Root', fname='master_fp.png')
#         plot_convergence(fp[5], title='False Position Convergence', fname='master_fp_conv.png')

#     if nr is not None and nr[0] is not None:
#         plot_with_root(f_np, 0, 3, root=nr[0], xl=xl, xu=xu,
#                         title='Newton-Raphson Root', fname='master_nr.png')
#         plot_convergence(nr[1], title='Newton-Raphson Convergence', fname='master_nr_conv.png')

#     # Newton-Raphson tangent geometry plot:
#     x0 = (xl + xu) / 2.0
#     plot_newton_raphson(x0=x0, a=0, b=3, title='NR Tangent Geometry', fname='master_nr_tangents.png')

# # -- STEP 3: Multi-root example (B1-style) -----------------------------------
# sth = 1.0
# f_multiroot_np = lambda x: 0.6*np.log(x+1) - sth*np.sin(1.7*x) - 0.08*x**2 - 0.08
# def f_multiroot(x):
#     return 0.6*math.log(x+1) - sth*math.sin(1.7*x) - 0.08*x**2 - 0.08
# _original_f = f                       # save before swapping
# globals()['f'] = f_multiroot          # swap global f for this example
# multi_intervals = scan_for_roots(0, 10, step=0.1)
# multi_roots = find_all_roots(0, 10, step=0.1, method='bisection', tol=0.0001)
# plot_with_brackets(f_multiroot_np, 0, 10, intervals=multi_intervals, roots=multi_roots,
#                     title='Section B: Multi-Root Scan', fname='master_multiroot.png')
# globals()['f'] = _original_f          # restore

# # -- STEP 4: Error/approximation theory demos --------------------------------
# print(f"\nMachine epsilon: {compute_machine_epsilon():.3e}")
# demo_taylor_truncation()
# demo_subtractive_cancellation()
# demo_error_tradeoff()