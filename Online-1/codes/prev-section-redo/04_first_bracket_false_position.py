"""
Section-C1: scan the domain, take the FIRST sign-change bracket only,
run False Position on it to ea <= 0.0001%, print the iteration table.

    f(x) = 2x^3 - 11.7x^2 + 17.7x - 5   on [0, 4], dx = 0.1
Expected first match: [0.3, 0.4]
"""
import numpy as np
from common import P, use, banner

A, B, STEP = 0.0, 4.0, 0.1
TOL = 0.0001


def f(x):
    return 2 * x ** 3 - 11.7 * x ** 2 + 17.7 * x - 5


def df(x):
    return 6 * x ** 2 - 23.4 * x + 17.7


def f_np(x):
    return 2 * x ** 3 - 11.7 * x ** 2 + 17.7 * x - 5


use(f, df)

banner('SECTION-C1: FIRST-MATCH BRACKET + FALSE POSITION')

direct, intervals = P.coarse_scan(A, B, STEP)
print(f'  Direct roots: {direct}')
print(f'  All sign-change intervals: {intervals}')

if not intervals:
    raise SystemExit('  No sign change found on [0, 4].')

xl, xu = intervals[0]                     # <- early exit: first match only
print(f'\n  FIRST MATCH: [{xl}, {xu}]   f({xl}) = {f(xl):.6f},  f({xu}) = {f(xu):.6f}')

res = P.false_position_opt(xl, xu, tol=TOL, verbose=True)
root, iters, ea, lu, uu, hist = res

print(f'\n  Root = {root:.8f}   ea = {ea:.8f}%   iterations = {iters}')
print(f'  xl updates = {lu},  xu updates = {uu}')
print('  ' + str(P.classify_and_verify(root, 'interval', residual_tol=1e-6)))

P.plot_with_root(f_np, A, B, root=root, xl=xl, xu=xu,
                 title='Section-C2: First Bracket, False Position',
                 fname='04_first_bracket_root.png')
P.plot_convergence(hist, title='False Position Convergence',
                   fname='04_falsepos_conv.png')
