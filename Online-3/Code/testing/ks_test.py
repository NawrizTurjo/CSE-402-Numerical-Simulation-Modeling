"""
Kolmogorov-Smirnov (K-S) test for uniformity on [0, 1).

H0: the sample comes from Uniform[0, 1).

Idea: compare the empirical CDF of the sorted sample against the
diagonal line y = x (the CDF of a true Uniform[0,1)). The test statistic
D is the single largest vertical gap between them, checked from both
above and below since the empirical CDF can overshoot or undershoot:

    sorted sample: R_(1) <= R_(2) <= ... <= R_(N)

    D+ = max_i ( i/N   - R_(i)   )     empirical CDF undershoots the diagonal
    D- = max_i ( R_(i) - (i-1)/N )     empirical CDF overshoots the diagonal
    D  = max(D+, D-)

Decision: reject H0 if D > D_critical(N, alpha).

scipy's `stats.kstest` (one-sample, against 'uniform') gives the same D
and an exact p-value for any N directly -- prefer it when you just need
an answer. The manual `ks_statistic` below is kept because some slide
questions want you to show the D+/D-/D table by hand.

Run standalone:
    python -m testing.ks_test
"""


import sys
from pathlib import Path

# Allow direct script execution from any directory
_CODE_DIR = Path(__file__).resolve().parent
while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
    _CODE_DIR = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from scipy import stats


def ks_statistic(sample):
    """Manual D+, D-, D computation (matches the slide's table layout)."""
    n = len(sample)
    s = sorted(sample)
    d_plus = max(i / n - s[i - 1] for i in range(1, n + 1))
    d_minus = max(s[i - 1] - (i - 1) / n for i in range(1, n + 1))
    return d_plus, d_minus, max(d_plus, d_minus)


def ks_uniform_test(sample, alpha=0.05):
    """
    Full K-S uniformity test. Uses scipy's `ksone` distribution for the
    critical value / p-value -- exact for any N and alpha (no table needed).
    """
    d_plus, d_minus, d = ks_statistic(sample)
    n = len(sample)
    d_critical = stats.ksone.ppf(1 - alpha, n)
    p_value = 1 - stats.ksone.cdf(d, n)
    # print(f"d_critical: {d_critical}")
    # print(f"p_value: {p_value}")
    # print(f"D value: {d}")
    decision = "Reject H0" if d > d_critical else "Do not reject H0"
    return {
        "D+": d_plus,
        "D-": d_minus,
        "D": d,
        "critical_value": d_critical,
        "p_value": p_value,
        "decision": decision,
    }


def plot_ks_test(sample, alpha=0.05, title="Kolmogorov-Smirnov Test (ECDF vs Theoretical)", show=True, save_path=None):
    """
    Plot Empirical CDF vs Theoretical Uniform(0,1) CDF with maximum gap D highlighted.

    Parameters
    ----------
    sample : list of float
        Sample variates.
    alpha : float, default=0.05
        Significance level.
    title : str, default='Kolmogorov-Smirnov Test (ECDF vs Theoretical)'
        Figure title.
    show : bool, default=True
        Whether to call plt.show().
    save_path : str, optional
        If provided, save the figure to this file path.
    """
    import matplotlib.pyplot as plt

    n = len(sample)
    s = sorted(sample)
    d_plus, d_minus, d = ks_statistic(sample)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Theoretical CDF y = x
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Theoretical CDF $F(x) = x$")

    # Empirical CDF step function
    x_steps = [0.0]
    y_steps = [0.0]
    for i, val in enumerate(s):
        x_steps.extend([val, val])
        y_steps.extend([i / n, (i + 1) / n])
    x_steps.append(1.0)
    y_steps.append(1.0)

    ax.plot(x_steps, y_steps, "b-", linewidth=2, label=f"Empirical CDF $S_n(x)$ (N={n})")
    ax.scatter(s, [(i + 1) / n for i in range(n)], color="blue", s=25, zorder=5)

    ax.set_title(f"{title}\n$D = {d:.4f}$ ($D^+={d_plus:.4f}, D^-={d_minus:.4f}$)", fontsize=12, fontweight="bold")
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("Cumulative Probability $F(x)$", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    # Slide's worked example.
    sample = [0.44, 0.81, 0.14, 0.05, 0.93]
    result = ks_uniform_test(sample)
    print(f"D+={result['D+']:.4f}  D-={result['D-']:.4f}  D={result['D']:.4f}")
    print(f"D_critical={result['critical_value']:.4f}  p={result['p_value']:.4f}  "
          f"-> {result['decision']}")
    # Expected (matches slide): D+=0.2600 D-=0.2100 D=0.2600 D_crit=0.5094 -> Do not reject H0
