"""
Section-B1: all real roots of
    f(x) = 0.6*ln(x+1) - C*sin(1.7x) - 0.08x^2 - 0.08
on 0 <= x <= 10 with incremental step dx = 0.1, solved by Bisection.
"""
import math
import numpy as np
from common import P, use, banner

C = 1.0
A, B, STEP = 0.0, 10.0, 0.1
TOL = 0.0001


def f(x):
    return 0.6 * math.log(x + 1) - C * math.sin(1.7 * x) - 0.08 * x ** 2 - 0.08


def df(x):
    return 0.6 / (x + 1) - C * 1.7 * math.cos(1.7 * x) - 0.16 * x


def f_np(x):
    return 0.6 * np.log(x + 1) - C * np.sin(1.7 * x) - 0.08 * x ** 2 - 0.08


use(f, df)

banner('SECTION-B1: MULTI-ROOT SCANNER (BISECTION)')

# --- Task 1: plot the whole domain first -----------------------------------
P.plot_standard(f_np, A, B, title='f(x) = 0.6ln(x+1) - sin(1.7x) - 0.08x^2 - 0.08',
                fname='03_function.png')

# --- Task 2: isolate every sign-change interval ----------------------------
direct, intervals = P.coarse_scan(A, B, STEP)
print(f'  Direct roots (|f| already ~ 0): {direct}')
print(f'  Sign-change intervals ({len(intervals)}): {intervals}')

# --- Tasks 3-4: bisect every bracket, one iteration table each -------------
roots = P.find_all_roots(A, B, step=STEP, method='bisection', tol=TOL)

print('\n  Verification (residual must be ~ 0 to ACCEPT):')
for r in roots:
    print('   ', P.classify_and_verify(r, 'interval', residual_tol=1e-4))

# --- Why a single global bisection on [0,10] would fail --------------------
print(f'\n  f(0) = {f(0):.6f},  f(10) = {f(10):.6f}  ->  f(0)*f(10) = {f(0)*f(10):.6f} > 0')
print('  Even root count cancels the global sign change: one bisection call on')
print('  [0,10] refuses to start, and even with an odd count it would converge')
print('  to exactly ONE root. Hence: scan first, then bisect each bracket.')

P.plot_with_brackets(f_np, A, B, intervals=intervals, roots=roots,
                     title='Section-B1: All Roots on [0,10]',
                     fname='03_all_roots.png')
