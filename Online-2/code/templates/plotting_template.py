"""
Plotting snippets -- copy the block you need
================================================
Not every exam asks for a plot, but if one does, these are the three shapes
that come up: convergence of an iterative method, a fitted curve against
data points, and a residual/error decay curve (log scale).

All snippets assume `plt.savefig(...)` works even without a display (the
exam machine may not have one) -- avoid plt.show() unless you know a
display is available; savefig always works headless.
"""
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# 1. Convergence of an eigenvalue estimate (power method / inverse power)
# ------------------------------------------------------------------
def plot_convergence(history, true_value=None, title="Convergence", fname="convergence.png"):
    iters = np.arange(1, len(history) + 1)
    plt.figure()
    plt.plot(iters, history, marker='o', label="estimate")
    if true_value is not None:
        plt.axhline(true_value, color='r', linestyle='--', label=f"true = {true_value:.4f}")
    plt.xlabel("iteration")
    plt.ylabel("eigenvalue estimate")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"saved -> {fname}")


# ------------------------------------------------------------------
# 2. Error / residual decay (log scale -- shows linear convergence as a
#    straight line, useful for "does this method converge linearly?")
# ------------------------------------------------------------------
def plot_error_decay(history, true_value, title="Error decay", fname="error_decay.png"):
    errors = [abs(h - true_value) for h in history]
    iters = np.arange(1, len(errors) + 1)
    plt.figure()
    plt.semilogy(iters, errors, marker='o')
    plt.xlabel("iteration")
    plt.ylabel("|estimate - true|  (log scale)")
    plt.title(title)
    plt.grid(True, which='both')
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"saved -> {fname}")


# ------------------------------------------------------------------
# 3. Fitted curve vs. data points (curve-fitting questions)
# ------------------------------------------------------------------
def plot_fit(t_data, y_data, evaluate_fn, title="Fitted curve", fname="fit.png"):
    t_dense = np.linspace(min(t_data), max(t_data), 200)
    y_dense = [evaluate_fn(t) for t in t_dense]
    plt.figure()
    plt.scatter(t_data, y_data, color='red', zorder=5, label="data")
    plt.plot(t_dense, y_dense, label="fitted curve")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"saved -> {fname}")


if __name__ == "__main__":
    # smoke test with fake data
    fake_history = [2.0, 2.8, 2.929, 2.976, 2.992, 3.0]
    plot_convergence(fake_history, true_value=3.0)
    plot_error_decay(fake_history, true_value=3.0)

    t = [5, 8, 12]
    y = [106.8, 177.2, 279.2]
    coeffs = np.array([0.290472, 19.6905, 1.08571])
    plot_fit(t, y, lambda tt: coeffs @ [tt**2, tt, 1])
