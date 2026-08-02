"""
Section-B2: pathological / discontinuous function

    f(x) = (x-2.5)^2 (x+1.5) / (x - 3.56)

Incremental scan + bisection on every bracket, then plot with the asymptote
masked. Two traps to demonstrate:
  * x = 2.5   -> double root, no sign change -> scanner MISSES it
  * x = 3.56  -> vertical asymptote, sign flips -> scanner finds a GARBAGE root
"""
import numpy as np
import matplotlib.pyplot as plt
from common import P, use, banner

A, B, STEP = -3.0, 5.0, 0.1
TOL = 0.0001
POLE = 3.56


def f(x):
    return ((x - 2.5) ** 2 * (x + 1.5)) / (x - POLE)


def f_np(x):
    return ((x - 2.5) ** 2 * (x + 1.5)) / (x - POLE)


use(f)

banner('SECTION-B2: PATHOLOGICAL RATIONAL FUNCTION (BISECTION)')

# --- Task 1: scan + bisect every bracket -----------------------------------
direct, intervals = P.coarse_scan(A, B, STEP)
print(f'  Direct roots: {direct}')
print(f'  Sign-change intervals: {intervals}')

roots = P.find_all_roots(A, B, step=STEP, method='bisection', tol=TOL)

print('\n  Verdicts (residual test is what separates real roots from garbage):')
verdicts = []
for r in roots:
    v = P.classify_and_verify(r, 'interval', residual_tol=1e-4)
    verdicts.append(v)
    tag = 'ACCEPT' if v['accepted'] else 'REJECT'
    print(f"   {tag}  x = {v['x']:.8f}   f(x) = {v['f_x']}")

real_roots = []
for v in verdicts:                        # keep accepted, drop near-duplicates
    if v['accepted'] and not any(abs(v['x'] - r) < 1e-4 for r in real_roots):
        real_roots.append(v['x'])

print('\n  Trap 1 - MISSED ROOT at x = 2.5:')
print(f"    f(2.4) = {f(2.4):.6f},  f(2.6) = {f(2.6):.6f}  ->  same sign.")
print('    (x-2.5)^2 has even multiplicity: the curve touches the axis but does')
print('    not cross it, so NO sign-change bracket can ever contain it.')
print(f'    It only shows up above because the grid (a={A}, step={STEP}) happens')
print('    to land exactly on 2.5, so the |f| <= tol "direct" test caught it.')
print('    Shift the grid off 2.5 and it vanishes. Real fix: a |f(x)| minimum')
print('    test (or Newton with m=2), never a sign test alone.')

print('\n  Trap 2 - GARBAGE ROOT at x = 3.56:')
print(f"    f(3.5) = {f(3.5):.6f},  f(3.6) = {f(3.6):.6f}  ->  sign flips.")
print('    That flip is a pole (f -> -inf then +inf), not a root. Bisection')
print('    converges happily onto it; only the residual check exposes it,')
print('    because |f| blows up instead of going to zero there.')

print(f'\n  TRUE roots kept: {[round(r, 8) for r in real_roots]}')

# --- Task 2: Re-use plot_with_brackets with a masked function ----------------
def f_masked(x):
    y = f_np(x)
    if isinstance(y, np.ndarray):
        y[np.abs(x - POLE) < 0.05] = np.nan
        y[np.abs(y) > 30] = np.nan
    return y

P.plot_with_brackets(
    f_masked,
    A,
    B,
    intervals=intervals,
    roots=real_roots,
    title='Section-B2: double root vs vertical asymptote',
    fname='05_pathological.png'
)
