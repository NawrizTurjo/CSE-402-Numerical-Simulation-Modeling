import numpy as np
import matplotlib.pyplot as plt

f = np.sin
f_prime_exact = np.cos

x0 = 1.0
true_slope = f_prime_exact(x0)

def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h

h_values = [1.0, 0.5, 0.1]

fig, axes = plt.subplots(1, len(h_values), figsize=(15, 4.5))

# Smooth curve for reference, plotted a bit wider than the h's we use
x_smooth = np.linspace(x0 - 0.5, x0 + 1.5, 400)
y_smooth = f(x_smooth)

for ax, h in zip(axes, h_values):
    approx_slope = forward_difference(f, x0, h)
    error = abs(true_slope - approx_slope)

    # the curve
    ax.plot(x_smooth, y_smooth, color="black", linewidth=2, label="f(x) = sin(x)")

    # the two points used by the forward-difference formula
    x1, x2 = x0, x0 + h
    y1, y2 = f(x1), f(x2)
    ax.plot([x1, x2], [y1, y2], "o", color="crimson", zorder=5)

    # secant line (approximation) extended across the plot for visibility
    x_line = np.array([x0 - 0.5, x0 + 1.5])
    secant_y = y1 + approx_slope * (x_line - x1)
    ax.plot(x_line, secant_y, "--", color="crimson", linewidth=2,
             label=f"secant (approx slope={approx_slope:.4f})")

    # true tangent line at x0
    tangent_y = y1 + true_slope * (x_line - x1)
    ax.plot(x_line, tangent_y, color="steelblue", linewidth=2,
             label=f"tangent (true slope={true_slope:.4f})")

    ax.set_title(f"h = {h}\nerror = {error:.5f}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_ylim(-0.2, 1.3)
    ax.legend(fontsize=8, loc="lower right")

fig.suptitle(f"Forward-Difference Approximation of f'(x) at x = {x0}  (true slope = {true_slope:.5f})",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("differentiation_plot.png", dpi=150)