"""
Chi-Square goodness-of-fit test for uniformity on [0, 1).

H0: the sample comes from Uniform[0, 1).
H1: it does not.

Procedure:
    1. Split [0, 1) into `bins` equal-width intervals.
    2. Count observed values O_i in each bin.
    3. Expected count per bin under H0: E_i = N / bins.
    4. Statistic: chi2 = sum_i (O_i - E_i)^2 / E_i     ~ chi-square(df = bins - 1)
    5. Decision: reject H0 if p-value < alpha (equivalently chi2 > critical value).

This module uses scipy (exact for ANY N, bins, alpha -- no lookup table
needed), which is what the C1 question explicitly permits ("You may use
Python's statistical libraries, such as SciPy"). A hand-computable,
library-free version of the statistic itself (step 1-4) is kept in
`chi_square_statistic_manual` in case a question asks you to show the
formula without scipy -- only the critical-value/p-value lookup needs
scipy or a printed table.

Run standalone:
    python -m testing.chi_square_test
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

from rng.common import bin_counts


def chi_square_statistic_manual(counts):
    """Step 1-4 without any library: chi2 = sum (O_i - E_i)^2 / E_i."""
    total = sum(counts)
    expected = total / len(counts)
    return sum(((observed - expected) ** 2) / expected for observed in counts)


def chi_square_uniform_test(uniforms, bins=10, alpha=0.05):
    """
    Full Chi-Square uniformity test on a list of values expected in [0, 1).

    Returns a dict with counts, the statistic, critical value, p-value,
    and the H0 decision -- this is exactly the row format the C1 table
    (Seed | Chi^2 | p-value | Decision) asks for.
    """
    counts = bin_counts(uniforms, bins)
    chi2 = chi_square_statistic_manual(counts)
    df = bins - 1
    critical_value = stats.chi2.ppf(1 - alpha, df=df)
    p_value = stats.chi2.sf(chi2, df=df)
    decision = "Reject H0" if p_value < alpha else "Do not reject H0"
    return {
        "counts": counts,
        "chi2": chi2,
        "df": df,
        "critical_value": critical_value,
        "p_value": p_value,
        "decision": decision,
    }


def plot_chi_square(uniforms, bins=10, alpha=0.05, title="Chi-Square Uniformity Test (Bin Frequencies)", show=True, save_path=None):
    """
    Plot observed bin counts against expected uniform count with Chi-Square test statistics.

    Parameters
    ----------
    uniforms : list of float
        Variates in [0, 1).
    bins : int, default=10
        Number of intervals.
    alpha : float, default=0.05
        Significance level.
    title : str, default='Chi-Square Uniformity Test (Bin Frequencies)'
        Figure title.
    show : bool, default=True
        Whether to call plt.show().
    save_path : str, optional
        If provided, save the figure to this file path.
    """
    import matplotlib.pyplot as plt

    result = chi_square_uniform_test(uniforms, bins=bins, alpha=alpha)
    counts = result["counts"]
    n = len(uniforms)
    expected = n / bins

    fig, ax = plt.subplots(figsize=(9, 5))
    bin_labels = [f"[{i/bins:.1f}, {(i+1)/bins:.1f})" for i in range(bins)]
    bars = ax.bar(bin_labels, counts, color="cornflowerblue", edgecolor="black", alpha=0.8, label="Observed $O_i$")
    ax.axhline(expected, color="crimson", linestyle="--", linewidth=2, label=f"Expected $E_i = {expected:.1f}$")

    # Add count labels on bars
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + max(counts) * 0.01, f"{int(h)}", ha="center", va="bottom", fontsize=9)

    ax.set_title(f"{title}\n$\\chi^2 = {result['chi2']:.4f}$ (Critical = {result['critical_value']:.4f}, p = {result['p_value']:.4g}) -> {result['decision']}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Interval Bins", fontsize=11)
    ax.set_ylabel("Observed Count", fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    import random

    random.seed(5731)
    sample = [random.random() for _ in range(1000)]
    result = chi_square_uniform_test(sample)
    print("Bin counts:", result["counts"])
    print(f"chi2={result['chi2']:.4f}  df={result['df']}  "
          f"critical={result['critical_value']:.4f}  p={result['p_value']:.6g}  "
          f"-> {result['decision']}")
    # print(f"chi2={result['chi2']:.4f}  df={result['df']}  "
    #           f"critical={result['critical_value']:.4f}  p={result['p_value']:.6g}  "
    #           f"-> {result['decision']}"
    #           f"\n\t->{result['decision2']}"
    # plot_chi_square(
    #     uniforms=sample,
    # )
