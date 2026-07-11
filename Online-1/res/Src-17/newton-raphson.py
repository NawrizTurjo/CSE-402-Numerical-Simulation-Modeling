import matplotlib.pyplot as plt
import numpy as np


def root_finder(
    f,
    guess,
    tolerance
):
    print(f"{'iter':^6} {'x':^6} {'x_new':^6} {'f(x_new)':^6} {'ea':^6}")
    x = guess
    ea = 1.0
    iter = 1
    while ea > tolerance:
        h = 1e-8
        df = (f(x + h) - f(x)) / h
        if df == 0:
            print("Newton-Raphson failed")
            return None
        x_new = x - (f(x) / df)

        if x_new == 0.0:
            print("Can't calculate Ea because x_new is zero")
            ea = float('inf')
        else: ea = abs((x_new - x) / x_new)

        print(f"{iter:^6} {x:^6.4f} {x_new:^6.4f} {f(x_new):^6.4f} {ea:^6.4f}")

        x = x_new
        iter += 1

    return x

def find_interval(f, start, end, step):
    x = start
    intervals = []
    while x < end:
        x_next = round(x + step, 6)
        if f(x) * f(x_next) < 0 or f(x) * f(x_next) == 0:
            intervals.append((round(x, 4), round(x_next, 4)))
        x = x_next
    return intervals

if __name__ == "__main__":
    f = lambda x: (x - 2) ** 3
    start = -3
    end = 3
    step = 0.5
    tol = 0.01

    x_vals = np.linspace(start, end, 100)
    y_vals = f(x_vals)
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals)
    # plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    # plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.grid(True)
    plt.show()

    intervals = find_interval(f, start, end, step)
    print(intervals)

    root = root_finder(f, 3, tol)
    print (root)
