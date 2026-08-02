"""
Loader: imports template.py (which contains the full prac-1 implementation)
and exposes it as P.

Usage in a solution file:
    from common import P, use, banner
    def f(x): ...
    def df(x): ...
    use(f, df)              # swap the globals template's solvers read
    P.bisection(xl, xu, tol=...)
"""
import os
import warnings
import matplotlib
matplotlib.use('Agg')          # headless: plt.show() becomes a no-op
warnings.filterwarnings("ignore", category=UserWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)

import template as P


def _safe_eval(x, func=None):
    """Replacement for P.safe_eval.

    The original signature is `safe_eval(x, func=f)` - the default is bound at
    def-time to the ORIGINAL f, so swapping P.f would not reach it. This one
    resolves P.f at call time instead. Same return contract: (value, None) on
    success, (None, error_message) on failure.
    """
    if func is None:
        func = P.f
    try:
        return func(x), None
    except (ZeroDivisionError, ValueError, OverflowError) as e:
        return None, str(e)


P.safe_eval = _safe_eval


def use(f, df=None):
    """Point template's solvers at this problem's function."""
    P.f = f
    if df is not None:
        P.df = df


def banner(title):
    print('\n' + '=' * 78)
    print(f'{title:^78}')
    print('=' * 78)
