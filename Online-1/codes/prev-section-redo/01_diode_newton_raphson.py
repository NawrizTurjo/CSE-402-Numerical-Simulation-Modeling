"""
Section-A1: Diode equation via Newton-Raphson.

    f(V) = Is*(exp(V/(n*VT)) - 1) + V/R - IL = 0
    f'(V) = Is/(n*VT) * exp(V/(n*VT)) + 1/R

V0 = 0.65 V, n = 1.8, R = 500 ohm, VT = 0.02585 V, IL = 0.2 mA, ea <= 0.0001%
Required headers: Iter | V_i | f(V_i) | f'(V_i) | V_{i+1} | ea (%)
"""
import math
import numpy as np
from common import P, use, banner

Is = 1e-12
n = 1.8
R = 500.0
VT = 0.02585
IL = 0.0002
V0 = 0.65
TOL = 0.0001

nVT = n * VT


def f(V):
    return Is * (math.exp(V / nVT) - 1.0) + V / R - IL


def df(V):
    return (Is / nVT) * math.exp(V / nVT) + 1.0 / R


use(f, df)

banner('SECTION-A1: DIODE EQUATION (NEWTON-RAPHSON)')
print(f'  n*VT = {nVT:.6f} V     Is = {Is:.1e} A     R = {R:.0f} ohm     IL = {IL} A')

# reuse prac-1's NR core, print the table in the exact required format
root, history = P._newton_raphson_core(V0, m=1, tol=TOL, max_iter=100, verbose=False)

print(f"\n{'Iter':<6}{'V_i':>14}{'f(V_i)':>18}{'f_prime(V_i)':>18}{'V_{i+1}':>16}{'ea (%)':>14}")
print('-' * 86)
for h in history:
    ea_str = '---' if h['iter'] == 1 else f"{h['ea']:.6f}"
    print(f"{h['iter']:<6}{h['xi']:>14.6f}{h['fxi']:>18.6e}{h['dfxi']:>18.6e}"
          f"{h['xi1']:>16.6f}{ea_str:>14}")
print('-' * 86)

print(f'\n  Operating voltage V ~ {root:.8f} V   after {len(history)} iterations')
print(f'  Residual f(V) = {f(root):.6e}')
print('  ' + str(P.classify_and_verify(root, 'newton_raphson', residual_tol=1e-9)))

# sanity: the first step is huge (ea ~ 511%) because f is exponentially stiff
# at V0 = 0.65 -> the tangent is far too shallow and overshoots down to ~0.106 V.

f_np = np.vectorize(f)
P.plot_with_root(f_np, 0.0, 0.9, root=root, title='Section-A1: Diode Operating Voltage', fname='01_diode.png')
