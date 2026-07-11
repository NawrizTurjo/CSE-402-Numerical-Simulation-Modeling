import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return x**3 - x - 1


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


def find_sign_change_intervals(a,b,h):
    intervals = []
    x1 = a
    


def false_position(x_l, x_u, es=0.001, max_iter=100):
    if f(x_l) * f(x_u) > 0:
        raise ValueError("Invalid interval: f(x_l) and f(x_u) must have opposite signs.")

    old_xr = None
    table = []

    for i in range(1, max_iter + 1):
        f_l = f(x_l)
        f_u = f(x_u)

        x_r = (x_u * f_l - x_l * f_u) / (f_l - f_u)
        f_r = f(x_r)

        if old_xr is None:
            ea = None
        else:
            ea = abs((x_r - old_xr) / x_r) * 100

        table.append((i, x_l, x_u, x_r, f_r, ea))

        if f_r == 0 or (ea is not None and ea <= es):
            return x_r, table
        
        
        # if f_r == 0 or (ea if not None and ea <= es):
        #     return x_r, table

        if f_l * f_r > 0:
            x_l = x_r
        else:
            x_u = x_r

        old_xr = x_r

    return x_r, table


def plot_graph():
    x = np.linspace(-2, 2.5, 600)
    y = [f(float(value)) for value in x]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="f(x) = x^3 - x - 1")
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.scatter([1, 2], [f(1), f(2)], color="red", zorder=3, label="Initial guesses")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Graph of f(x) = x^3 - x - 1")
    plt.grid(True)
    plt.legend()
    plt.savefig("graph.png", dpi=200, bbox_inches="tight")
    plt.show()


def print_table(table):
    print("iter       x_l       x_u       x_r       f(x_r)      ea(%)")
    for i, x_l, x_u, x_r, f_r, ea in table:
        if ea is None:
            print(f"{i:4d}  {x_l:8.5f}  {x_u:8.5f}  {x_r:8.5f}  {f_r:10.5f}       ---")
        else:
            print(f"{i:4d}  {x_l:8.5f}  {x_u:8.5f}  {x_r:8.5f}  {f_r:10.5f}  {ea:8.5f}")


def main():
    x_l = 1
    x_u = 2

    print("Equation: x^3 - x - 1 = 0")
    print("f(x) = x^3 - x - 1")
    print()

    print(f"f({x_l}) = {f(x_l):.6f}")
    print(f"f({x_u}) = {f(x_u):.6f}")
    print("Since f(1)*f(2) < 0, choose x_l = 1 and x_u = 2.")
    print()

    plot_graph()

    root, table = false_position(x_l, x_u, es=0.001)

    print("False position iteration table:")
    print_table(table)
    print(f"\nApproximate root = {root:.6f}")

    print("\nAll real roots:")
    print(f"{root:.6f}")


if __name__ == "__main__":
    main()
