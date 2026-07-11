"""
Numerical Analysis Online Exam Python Cheat Sheet

Use this file to quickly remember common math syntax, error formulas,
root-finding formulas, and Matplotlib plotting code.
"""

import math

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. BASIC MATH SYNTAX
# ============================================================

x = 2.5

# Power
x_squared = x**2
x_cubed = x**3
x_half_power = x**0.5
x_two_third_power = x ** (2 / 3)

# Square root
sqrt_x = math.sqrt(x)

# Exponential and logarithm
e_power_x = math.exp(x)       # e^x
natural_log = math.log(x)     # ln(x)
log_base_10 = math.log10(x)   # log10(x)

# Trigonometry: input must be in radians
sin_x = math.sin(x)
cos_x = math.cos(x)
tan_x = math.tan(x)

# Convert degree to radian
angle_degree = 30
angle_radian = math.radians(angle_degree)
sin_30 = math.sin(angle_radian)

# Constants
pi_value = math.pi
e_value = math.e

# Absolute value
absolute_value = abs(-7.2)

# Rounding
rounded_value = round(1.23456789, 4)  # 1.2346


# ============================================================
# 2. COMMON FUNCTION WRITING STYLE
# ============================================================

def f_example(x):
    return x**3 - x - 1


def df_example(x):
    return 3 * x**2 - 1


# Example with ln, sin, and power:
def f_section_b(x):
    sth = 1.0
    return 0.6 * math.log(x + 1) - sth * math.sin(1.7 * x) - 0.08 * x**2 - 0.08


# Example with pi:
def f_dipstick(h):
    r = 4
    V = 5
    return math.pi * h**2 * (3 * r - h) / 3 - V


def df_dipstick(h):
    r = 4
    return math.pi * (2 * r * h - h**2)


# ============================================================
# 3. ERROR FORMULAS
# ============================================================

def approximate_relative_error(new_value, old_value):
    return abs((new_value - old_value) / new_value) * 100


def true_relative_error(true_value, approximate_value):
    return abs((true_value - approximate_value) / true_value) * 100


# ============================================================
# 4. SIGN CHECK AND INTERVAL SCAN
# ============================================================

def has_sign_change(func, a, b):
    return func(a) * func(b) < 0


def find_sign_change_intervals(func, a, b, h):
    intervals = []
    x1 = a
    f1 = func(x1)

    while x1 < b:
        x2 = round(x1 + h, 10)
        if x2 > b:
            x2 = b

        f2 = func(x2)

        if f1 == 0:
            intervals.append((x1, x1))
        elif f1 * f2 < 0:
            intervals.append((x1, x2))

        x1 = x2
        f1 = f2

    return intervals


# ============================================================
# 5. BISECTION FORMULA
# ============================================================

def bisection_formula(x_l, x_u):
    return (x_l + x_u) / 2


# ============================================================
# 6. FALSE POSITION FORMULA
# ============================================================

def false_position_formula(func, x_l, x_u):
    f_l = func(x_l)
    f_u = func(x_u)
    return (x_u * f_l - x_l * f_u) / (f_l - f_u)


# ============================================================
# 7. NEWTON-RAPHSON FORMULA
# ============================================================

def newton_raphson_formula(func, dfunc, x_i):
    return x_i - func(x_i) / dfunc(x_i)


# ============================================================
# 8. QUICK ROOT-FINDING BASE CODES
# ============================================================

def bisection_method(func, x_l, x_u, es=0.001, max_iter=100):
    if func(x_l) * func(x_u) > 0:
        print("Invalid interval")
        return None

    old_mid = None

    for i in range(1, max_iter + 1):
        mid = (x_l + x_u) / 2
        f_mid = func(mid)

        if old_mid is None:
            ea = None
        else:
            ea = abs((mid - old_mid) / mid) * 100

        if f_mid == 0 or (ea is not None and ea <= es):
            return mid

        if func(x_l) * f_mid < 0:
            x_u = mid
        else:
            x_l = mid

        old_mid = mid

    return mid


def false_position_method(func, x_l, x_u, es=0.001, max_iter=100):
    if func(x_l) * func(x_u) > 0:
        print("Invalid interval")
        return None

    old_xr = None

    for i in range(1, max_iter + 1):
        f_l = func(x_l)
        f_u = func(x_u)

        x_r = (x_u * f_l - x_l * f_u) / (f_l - f_u)
        f_r = func(x_r)

        if old_xr is None:
            ea = None
        else:
            ea = abs((x_r - old_xr) / x_r) * 100

        if f_r == 0 or (ea is not None and ea <= es):
            return x_r

        if f_l * f_r > 0:
            x_l = x_r
        else:
            x_u = x_r

        old_xr = x_r

    return x_r


def newton_raphson_method(func, dfunc, x0, es=0.001, max_iter=100):
    old_x = x0

    for i in range(1, max_iter + 1):
        if dfunc(old_x) == 0:
            print("Derivative is zero")
            return None

        new_x = old_x - func(old_x) / dfunc(old_x)
        ea = abs((new_x - old_x) / new_x) * 100

        if ea <= es:
            return new_x

        old_x = new_x

    return new_x


# ============================================================
# 9. PRINTING TABLES
# ============================================================

def print_table_example():
    i = 1
    x_l = 1
    x_u = 2
    x_r = 1.1666667
    f_r = -0.5787037

    print("iter       x_l       x_u       x_r       f(x_r)")
    print(f"{i:4d}  {x_l:8.5f}  {x_u:8.5f}  {x_r:8.5f}  {f_r:10.5f}")


# Formatting reminder:
# {i:4d}      integer, total width 4
# {x:8.5f}   float, total width 8, 5 digits after decimal
# {x:10.6f}  float, total width 10, 6 digits after decimal


# ============================================================
# 10. MATPLOTLIB IMPORTANT CODES
# ============================================================

def plot_basic_function():
    x_values = np.linspace(-2, 3, 500)
    y_values = [f_example(float(x)) for x in x_values]

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, label="f(x) = x^3 - x - 1")
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Graph of f(x)")
    plt.grid(True)
    plt.legend()
    plt.savefig("graph.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_with_sign_check():
    x_values = np.linspace(-2, 3, 500)
    y_values = [f_example(float(x)) for x in x_values]

    x_l = 1
    x_u = 2

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, label="f(x)")
    plt.axhline(0, color="black", linewidth=1)
    plt.scatter([x_l, x_u], [f_example(x_l), f_example(x_u)], color="red", label="Initial guesses")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Sign Check on Graph")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_multiple_functions():
    x_values = np.linspace(-2, 2, 500)
    y1 = [x**2 for x in x_values]
    y2 = [x**3 for x in x_values]

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y1, label="x^2")
    plt.plot(x_values, y2, label="x^3")
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Multiple Functions")
    plt.grid(True)
    plt.legend()
    plt.show()


# ============================================================
# 11. NUMPY QUICK SYNTAX
# ============================================================

# Create evenly spaced values:
values_1 = np.linspace(0, 10, 101)

# Create values with step size:
values_2 = np.arange(0, 10.1, 0.1)

# Apply function to NumPy array:
values_y = values_1**2 + 3 * values_1 - 5


# ============================================================
# 12. COMMON EXAM EXAMPLES
# ============================================================

def example_false_position_problem():
    root = false_position_method(f_example, 1, 2, es=0.001)
    print(f"Root = {root:.6f}")


def example_newton_dipstick_problem():
    root = newton_raphson_method(f_dipstick, df_dipstick, 0.5, es=0.05)
    print(f"h = {root:.6f}")


if __name__ == "__main__":
    print("Numerical Analysis Cheat Sheet")
    print("Run individual functions from this file when needed.")
    print()
    print("Example:")
    example_false_position_problem()
