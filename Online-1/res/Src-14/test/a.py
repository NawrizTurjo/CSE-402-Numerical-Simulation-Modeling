import math

import matplotlib.pyplot as plt
import numpy as np


# Change this value if the exam gives a different sth.
sth = 1.0


def f(x):
    return 0.6 * math.log(x + 1) - sth * math.sin(1.7 * x) - 0.08 * x**2 - 0.08


def find_sign_change_intervals(a, b, h):
    intervals = []
    x1 = a
    f1 = f(x1)

    while x1 < b:
        x2 = round(x1 + h, 10)
        if x2 > b:
            x2 = b

        f2 = f(x2)

        if f1 == 0:
            intervals.append((x1, x1))
        elif f1 * f2 < 0:
            intervals.append((x1, x2))

        x1 = x2
        f1 = f2

    return intervals


def bisection(a, b, tol=1e-6, max_iter=100):
    if a == b:
        return a, [(1, a, b, a, f(a))]

    fa = f(a)
    fb = f(b)

    if fa * fb > 0:
        raise ValueError("Bisection needs f(a) and f(b) to have opposite signs.")

    table = []

    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        table.append((i, a, b, c, fc))

        if abs(fc) < tol or abs(b - a) / 2 < tol:
            return c, table

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return c, table


def plot_graph():
    xs = np.linspace(0, 10, 1000)
    ys = [f(float(x)) for x in xs]

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, label="f(x)")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Graph of f(x) on [0, 10]")
    plt.grid(True)
    plt.legend()
    plt.savefig("graph.png", dpi=200, bbox_inches="tight")
    plt.show()


def print_table(table):
    print("iter         a         b         mid       f(mid)")
    for i, a, b, c, fc in table:
        print(f"{i:4d}  {a:8.5f}  {b:8.5f}  {c:8.5f}  {fc:11.6f}")


def main():
    a = 0
    b = 10
    h = 0.1
    tol = 1e-6

    plot_graph()

    intervals = find_sign_change_intervals(a, b, h)

    print("Sign-change intervals with step size h = 0.1:")
    for left, right in intervals:
        print(f"[{left:.1f}, {right:.1f}]")

    print("\nBisection results:")
    roots = []

    for k, (left, right) in enumerate(intervals, start=1):
        root, table = bisection(left, right, tol)
        roots.append(root)

        print(f"\nRoot {k}")
        print(f"Initial interval: [{left:.1f}, {right:.1f}]")
        print_table(table)
        print(f"Approximate root = {root:.6f}")

    print("\nAll roots:")
    for root in roots:
        print(f"{root:.6f}")

    print("\nWhy not bisection once on [0, 10]?")
    print("Bisection needs f(0)*f(10) < 0. If the endpoint signs are same,")
    print("it cannot start. Also, one bisection run can find only one root,")
    print("so it may miss other roots inside [0, 10].")


if __name__ == "__main__":
    main()
