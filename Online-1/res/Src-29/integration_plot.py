import numpy as np
import matplotlib.pyplot as plt

f = np.sin
a, b = 0, np.pi   # integration interval

def riemann_sum(f, a, b, n):
    x = np.linspace(a, b, n, endpoint=False)  # left endpoints of each rectangle
    width = (b - a) / n
    return np.sum(f(x) * width), x, width

true_integral = -np.cos(b) + np.cos(a)  # exact value = 2

# Show how the approximation improves as n (number of rectangles) increases
n_values = [5, 10, 30]

fig, axes = plt.subplots(1, len(n_values), figsize=(15, 4.5))

x_smooth = np.linspace(a, b, 400)
y_smooth = f(x_smooth)

for ax, n in zip(axes, n_values):
    approx, x_left, width = riemann_sum(f, a, b, n)
    error = abs(true_integral - approx)

    # smooth curve
    ax.plot(x_smooth, y_smooth, color="black", linewidth=2)

    # rectangles
    ax.bar(x_left, f(x_left), width=width, align="edge",
           alpha=0.5, edgecolor="black", color="steelblue")

    ax.set_title(f"n = {n}\napprox = {approx:.5f}, error = {error:.5f}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x) = sin(x)")
    ax.set_ylim(0, 1.15)

fig.suptitle(f"Riemann Sum Approximation of ∫ sin(x) dx from 0 to π  (true value = {true_integral:.5f})",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("riemann_sum_plot.png", dpi=150)