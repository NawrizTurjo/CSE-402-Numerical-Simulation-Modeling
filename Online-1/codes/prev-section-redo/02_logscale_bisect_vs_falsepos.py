"""
Section-C2 : f(x) = ln(x) on [1e-4, 1e4], ea <= 0.0001%.
Log-x plot, sign-change check, Bisection vs False Position, update counts,
full iteration tables + final comparison table.
"""
import math
import numpy as np
from common import P, use, banner

XL, XU = 1e-4, 1e4
TOL = 0.0001


def f(x):
    return math.log(x)


def df(x):
    return 1.0 / x


use(f, df)

banner('SECTION-C2: ln(x) ON [1e-4, 1e4]')

# --- Task 2: verify the sign change on the boundaries ----------------------
fl, fu = f(XL), f(XU)
print(f'  f(xl) = f({XL:.0e}) = {fl:.6f}')
print(f'  f(xu) = f({XU:.0e}) = {fu:.6f}')
print(f'  f(xl)*f(xu) = {fl*fu:.6f} < 0  ->  sign change, root bracketed\n')

# --- Task 1: log-scale plot ------------------------------------------------
P.plot_logscale(np.log, XL, XU, title='f(x) = ln(x)  (log x-axis)',
                fname='02_lnx_logscale.png')

# --- Tasks 3-5: run both methods, full tables ------------------------------
bi = P.bisection(XL, XU, tol=TOL, max_iter=500, verbose=True)
fp = P.false_position_opt(XL, XU, tol=TOL, max_iter=500, verbose=True)

# --- Task 6: comparison table ---------------------------------------------
print(f"\n{'='*92}")
print(f"{'COMPARISON TABLE':^92}")
print(f"{'='*92}")
print(f"{'Method':<18}{'Root':>16}{'Iterations':>13}{'Final f(xr)':>18}"
      f"{'xl updates':>13}{'xu updates':>13}")
print('-' * 92)
for name, res in (('Bisection', bi), ('False Position', fp)):
    if res is None:
        print(f'{name:<18}{"TERMINATED (undefined value hit)":>74}')
        continue
    root, iters, ea, lu, uu, hist = res
    print(f'{name:<18}{root:>16.8f}{iters:>13}{f(root):>18.8e}{lu:>13}{uu:>13}')
print('=' * 92)

if fp is not None and (fp[3] == 0 or fp[4] == 0):
    stuck = 'xl' if fp[3] == 0 else 'xu'
    print(f'\n  [STAGNATION] False Position: {stuck} never moved.')
    print('  ln(x) is concave down (f\'\' = -1/x^2 < 0), so every secant crosses')
    print('  the axis to the RIGHT of the true root x=1 -> f(xr)>0 -> xu=xr each')
    print(f'  time, xl frozen at {XL:.0e}. Bracketed method degrades to a slow')
    print('  one-sided open method: hence far more iterations than bisection.')

# --- convergence curves ----------------------------------------------------
if bi is not None:
    P.plot_convergence(bi[5], title='Bisection Convergence (ln x)',
                       fname='02_bisection_conv.png')
if fp is not None:
    P.plot_convergence(fp[5], title='False Position Convergence (ln x)',
                       fname='02_falsepos_conv.png')
