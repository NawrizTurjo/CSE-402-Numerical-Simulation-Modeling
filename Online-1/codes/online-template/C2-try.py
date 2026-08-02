"""
=============================================================================
  CSE-402 NUMERICAL LAB EXAM STARTER TEMPLATE
=============================================================================
Exam Instructions:
  1. Define your problem's f(x), df(x), f_np(x) in Section (1).
  2. Call `use(f, df)` to register them with template solvers.
  3. Uncomment/run whichever exam section matches your question in Section (2).
=============================================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from common import P, use, banner

# =============================================================================
# (1) DEFINE YOUR QUESTION'S FUNCTION & DERIVATIVE HERE
# =============================================================================

def f(x):
    # --- Example 1: Standard Polynomial (e.g. x^3 - x - 1) ---
    # return x**3 - x - 1

    # --- Example 2: Transcendental (e.g. 0.6*ln(x+1) - sin(1.7x) - 0.08x^2 - 0.08) ---
    # return 0.6 * math.log(x + 1) - math.sin(1.7 * x) - 0.08 * x**2 - 0.08

    # --- Example 3: Diode Voltage Equation ---
    # nVT = 1.8 * 0.02585
    # return 1e-12 * (math.exp(x / nVT) - 1.0) + x / 500.0 - 0.0002

    # --- Example 4: Pathological Rational ---
    # return ((x - 2.5)**2 * (x + 1.052)) / (x - 3.551)
    return np.log(x)


def df(x):
    # Derivative of f(x) (Needed for Newton-Raphson)
    # return 3 * x**2 - 1

    # Numeric central difference fallback if f'(x) is hard to derive:
    # h = 1e-5
    # return (f(x + h) - f(x - h)) / (2 * h)

    return 1.0 / x


def f_np(f):
    # Vectorized version for numpy / matplotlib plotting
    return np.vectorize(f)


# Connect functions to master template solvers
use(f, df)

a = 1e-4
b = 1e4


# =============================================================================
# (2) EXAM WORKFLOWS - UNCOMMENT THE WORKFLOW YOU NEED
# =============================================================================

if __name__ == '__main__':
    banner("EXAM EXECUTION STARTER")



    P.plot_logscale(
        f_np=f_np(f),
        a = a,
        b = b,
        title='log-scale plot for $ln(x)$',
        fname = 'fig/try-c2-graph-plot'
    )

    fxl,_ = P.safe_eval(a)
    fxu,_ = P.safe_eval(b)

    mult_val = fxl * fxu

    print(f"fxl = {fxl:.6f} and fxu = {fxu:.6f} \nfxl * fxu = {mult_val:.6f}\nso, the sign changes as mult_val < 0 is {mult_val < 0}")

    bi = P.bisection(a,b)
    fp = P.false_position_opt(a,b)

    a = 20+10+16+18+10+10
    print('='*a)
    print(f"{'Comparison Table':^{a}}")
    # print(a)
    print('-'*a)
    print(f"{'Method':20}{'Root':>10}{'Iterations':>16}{'Final f(x_r)':>18}{'xl cnt':>10}{'xu cnt':>10}")

    root, iterations, _, xls, xus, _ = bi
    name = 'Bisection'
    final_fxr = f(root)
    print(f"{name:20}{root:>10.8f}{iterations:>16}{final_fxr:>18.6e}{xls:>10}{xus:>10}")
    root, iterations, _, xls, xus, _ = fp
    name = 'False-Position'
    final_fxr = f(root)
    print(f"{name:20}{root:>10.8f}{iterations:>16}{final_fxr:>18.6e}{xls:>10}{xus:>10}")
    print('='*a)

    P.plot_convergence(
        bi[-1],
        title="bisection convergence of $ln(x)$",
        fname="fig/lnx_bisection_conv.png"
    )

    P.plot_convergence(
        fp[-1],
        title="false position convergence of $ln(x)$",
        fname="fig/lnx_fp_conv.png"
    )

    # -------------------------------------------------------------------------
    # WORKFLOW A: Quick Interval Scan & Candidate Classification
    # -------------------------------------------------------------------------
    # a, b, step = 0.0, 5.0, 0.1
    # direct, intervals = P.coarse_scan(a, b, step, direct_tol=1e-8)
    # print(f"Direct candidates (|f| <= 1e-8): {direct}")
    # print(f"Sign-change intervals ({len(intervals)}): {intervals}")


    # -------------------------------------------------------------------------
    # WORKFLOW B: Bisection Method (Single Bracket or All Brackets)
    # -------------------------------------------------------------------------
    # --- Single Bracket Run ---
    # xl, xu, tol = 1.0, 2.0, 0.0001
    # res = P.bisection(xl, xu, tol=tol, verbose=True)
    # if res is not None:
    #     root, i, ea, lu, uu, hist = res
    #     print('  Verdict:', P.classify_and_verify(root, 'bisection'))
    #     P.plot_with_root(f_np, a, b, root=root, xl=xl, xu=xu, title="Bisection Root", fname="exam_bisect.png")
    #     P.plot_convergence(hist, title="Bisection Error Decay", fname="exam_bisect_conv.png")

    # --- Multi-Root Run (All Brackets) ---
    # roots = P.find_all_roots(a, b, step=step, method='bisection', tol=0.0001)
    # P.plot_with_brackets(f_np, a, b, intervals=intervals, roots=roots, title="All Roots Scan", fname="exam_multiroot.png")


    # -------------------------------------------------------------------------
    # WORKFLOW C: False Position (Regula Falsi / Illinois Anti-Stagnation)
    # -------------------------------------------------------------------------
    # --- Standard / Optimized False Position ---
    # xl, xu, tol = 1.0, 2.0, 0.0001
    # res = P.false_position_opt(xl, xu, tol=tol, verbose=True)
    # if res is not None:
    #     root, i, ea, lu, uu, hist = res
    #     print(f"xl updates: {lu}, xu updates: {uu}")
    #     print('  Verdict:', P.classify_and_verify(root, 'false_position'))

    # --- Illinois Variant (Halves weight on stagnant bound) ---
    # res = P.false_position_illinois(xl, xu, tol=tol, verbose=True)


    # -------------------------------------------------------------------------
    # WORKFLOW D: Newton-Raphson Method
    # -------------------------------------------------------------------------
    # x0, tol = 1.5, 0.0001
    # --- Analytic / Numeric Derivative NR ---
    # root, hist = P.newton_raphson(x0, tol=tol, verbose=True)
    # print('  Verdict:', P.classify_and_verify(root, 'newton_raphson'))
    # P.plot_newton_raphson(x0, a, b, title="Newton-Raphson Tangents", fname="exam_nr_tangents.png")

    # --- Multiple Roots (Multiplicity factor m) ---
    # root, hist = P.advanced_newton_raphson(x0, m=2, tol=tol, verbose=True)


    # -------------------------------------------------------------------------
    # WORKFLOW E: Side-by-Side Comparison (Bisection vs False Position vs NR)
    # -------------------------------------------------------------------------
    # xl, xu, tol = 1.0, 2.0, 0.0001
    # bi, fp, nr = P.compare_methods(xl, xu, tol=tol)


    # -------------------------------------------------------------------------
    # WORKFLOW F: Pathological Rational Function (Asymptote & Residual Check)
    # -------------------------------------------------------------------------
    # POLE = 3.551
    # def f_masked(x):
    #     y = f_np(x)
    #     if isinstance(y, np.ndarray):
    #         y[np.abs(x - POLE) < 0.05] = np.nan
    #         y[np.abs(y) > 30] = np.nan
    #     return y
    # P.plot_with_brackets(f_masked, -2, 5, intervals=intervals, roots=roots, title="Pathological Function", fname="exam_pathological.png")
