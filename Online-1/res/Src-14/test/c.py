import math

import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 1000 - 298 * x + 3 * x ** (2 / 3)

def df(x):
    return -298 + 2 * x ** (-1 / 3)

# def newton_raphson(x0, es=0.05, max_iter=50):
#     rows = []
#     old_x = x0
    
#     for i in range(1, max_iter + 1):
#         new_x = old_x - f(old_x) / df(old_x)
#         ea = abs((new_x - old_x) / new_x) * 100
#         rows.append((i, new_x, f(new_x), ea))
        
#         if ea <= es:
#             return new_x, rows
        
#         old_x = new_x   
        
        
#         if ea <= es:
#             return new_x, rows
        
#         old_x = new_x
        
#     return new_x, rows
        

def newton_raphson(x0, es=0.05, max_iter=50):
    rows = []
    old_x = x0

    for i in range(1, max_iter + 1):
        new_x = old_x - f(old_x) / df(old_x)
        ea = abs((new_x - old_x) / new_x) * 100
        rows.append((i, new_x, f(new_x), ea))

        if ea <= es:
            return new_x, rows

        old_x = new_x

    return new_x, rows


def plot_graph():
    x = np.linspace(0.1, 8, 600)
    y = [f(float(value)) for value in x]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="f(x) = 1000 - 298x + 3x^(2/3)")
    plt.axhline(0, color="black", linewidth=1)
    plt.scatter([3, 4], [f(3), f(4)], color="red", zorder=3, label="Sign check")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Break-even Function")
    plt.grid(True)
    plt.legend()
    plt.savefig("graph.png", dpi=200, bbox_inches="tight")
    plt.show()
    
def plot_graph():
    x = np.linspace(0.1, 8, 600)


def main():
    x0 = 4
    es = 0.05

    print("Cost: C(x) = 1000 + 2x + 3x^(2/3)")
    print("Revenue: R(x) = 300x")
    print("Break-even equation: C(x) = R(x)")
    print("f(x) = 1000 - 298x + 3x^(2/3)")
    print("f'(x) = -298 + 2x^(-1/3)")
    print()

    print(f"f(3) = {f(3):.6f}")
    print(f"f(4) = {f(4):.6f}")
    print("Since f(3) and f(4) have opposite signs, choose x0 = 4.")
    print()

    root, rows = newton_raphson(x0, es)

    print("Newton-Raphson iteration table:")
    print("iter      x estimate          f(x)        ea (%)")
    for iteration, x, fx, ea in rows:
        print(f"{iteration:4d}  {x:14.6f}  {fx:12.6f}  {ea:12.6f}")

    print()
    print(f"Break-even quantity = {root:.6f} grams per day")

    plot_graph()


if __name__ == "__main__":
    main()

