"""
C2 (online exam, section C2, see Res/questions.md):
When the Generator Fails the Test: Hidden Period Collapse and the K-S Test.

Problem:
Investigate a Multiplicative Congruential Generator (MCG):
    X_{n+1} = (a * X_n) mod m,    U_n = X_n / m
with m = 2^16 = 65536, a = 5, X0 = 1 (c = 0).

Key Insight:
Because c = 0, the Hull-Dobell full-period conditions do not apply.
For m = 2^b, the theoretical max period is m/4 = 16,384 (achieved here
since X0 is odd and a = 5 = 8*0 + 5). When drawing N = 100,000 samples,
the generator repeats its cycle ~6.1 times (p = 16,384 << N = 100,000).
Yet, the one-sample Kolmogorov-Smirnov (K-S) test FAILS TO REJECT H0
because K-S tests only the 1-D marginal distribution of sorted values,
which remains uniform, completely blind to the exact periodic collapse.

Run standalone:
    python -m solutions.c2_hidden_period_collapse
"""


import sys
from pathlib import Path

# Allow direct script execution from any directory
_CODE_DIR = Path(__file__).resolve().parent
while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
    _CODE_DIR = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import math

from rng.lcg import find_lcg_cycle, check_max_period_conditions
from testing.ks_test import ks_statistic


def lcg_generator(n_samples, m=65536, a=5, X0=1):
    """
    Multiplicative Congruential Generator (MCG) with c = 0.

    Parameters
    ----------
    n_samples : int
        Number of variates to generate (N).
    m : int, default=65536 (2^16)
        Modulus.
    a : int, default=5
        Multiplier.
    X0 : int, default=1
        Initial seed.

    Returns
    -------
    list of float
        Generated sequence of U_n = X_n / m in [0, 1).
    """
    uniforms = []
    x = X0
    for _ in range(n_samples):
        x = (a * x) % m
        uniforms.append(x / m)
    return uniforms


def perform_ks_test(sample, alpha=0.05):
    """
    Perform a one-sample, two-sided Kolmogorov-Smirnov (K-S) test against Uniform(0, 1).
    Follows the 5-step procedure specified in Task 2.

    Parameters
    ----------
    sample : list of float
        Sample variates.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    dict
        D+, D-, D_N, D_crit, decision, and sample size N.
    """
    n = len(sample)
    d_plus, d_minus, d_n = ks_statistic(sample)
    # Large-sample critical value threshold: 1.36 / sqrt(N) for alpha=0.05
    d_crit = 1.36 / math.sqrt(n)
    decision = "Reject H0" if d_n > d_crit else "Fail to reject H0 (Accept H0)"

    return {
        "N": n,
        "D+": d_plus,
        "D-": d_minus,
        "D_N": d_n,
        "D_crit": d_crit,
        "alpha": alpha,
        "decision": decision,
    }


def task1_generate():
    """Task 1: Generate a small sample to demonstrate lcg_generator."""
    print("=" * 70)
    print("TASK 1: MCG Stream Generator")
    print("=" * 70)
    samples = lcg_generator(10, m=65536, a=5, X0=1)
    print("First 10 generated U_n values (m=65536, a=5, X0=1):")
    for i, u in enumerate(samples, start=1):
        print(f"  U_{i:<2d} = {u:.6f}  (X_{i} = {int(round(u * 65536))})")
    print()
    return samples


def task2_ks_test():
    """Task 2: Generate N=100,000 samples and perform 2-sided K-S test."""
    print("=" * 70)
    print("TASK 2: Two-Sided Kolmogorov-Smirnov (K-S) Uniformity Test")
    print("=" * 70)
    n_samples = 100_000
    print(f"1. Generating N = {n_samples:,} samples using lcg_generator...")
    stream = lcg_generator(n_samples, m=65536, a=5, X0=1)

    print("2. Sorting variates in ascending order: U_(1) <= ... <= U_(N)...")
    print("3. Computing empirical CDF and test statistic D_N = max(D+, D-)...")
    result = perform_ks_test(stream, alpha=0.05)

    print("4. Evaluating critical threshold D_crit = 1.36 / sqrt(N)...")
    print(f"   D+                  = {result['D+']:.6f}")
    print(f"   D-                  = {result['D-']:.6f}")
    print(f"   D_N (KS Statistic)  = {result['D_N']:.6f}")
    print(f"   D_crit (alpha=0.05) = {result['D_crit']:.6f}  (1.36 / sqrt({n_samples}))")
    print()
    print(f"5. Decision Rule: D_N ({result['D_N']:.6f}) {'<' if result['D_N'] <= result['D_crit'] else '>'} D_crit ({result['D_crit']:.6f})")
    print(f"   -> Result: {result['decision']}")
    print(f"   -> Conclusion: Marginal distribution appears consistent with Uniform(0, 1).")
    print()
    return result


def task3_period_and_reflection():
    """Task 3: Calculate period programmatically and answer conceptual questions."""
    print("=" * 70)
    print("TASK 3: Period Calculation & Structural Analysis")
    print("=" * 70)

    # 3(a) Programmatic period calculation
    m, a, c, X0 = 65536, 5, 0, 1
    repeated_val, repeat_iter, measured_period = find_lcg_cycle(X0, a, c, m)
    case, theoretical_max_p, satisfied, details = check_max_period_conditions(a, c, m, X0)

    print("(a) Programmatic vs Theoretical Period:")
    print(f"    - Programmatically measured period : p = {measured_period}")
    print(f"    - First repeated value             : {repeated_val} at iteration {repeat_iter}")
    print(f"    - Theoretical condition            : {case}")
    print(f"    - Theoretical max period           : m / 4 = 65536 / 4 = {theoretical_max_p}")
    print(f"    - Condition satisfied              : {satisfied} ({details})")
    print(f"    - Match confirmed                  : {measured_period == theoretical_max_p}")
    print()

    # 3(b) Detailed explanation
    print("(b) Concrete behavior for n > p and why K-S test fails to detect it:")
    print("-" * 70)
    print(
        "1. What happens for n > p (n > 16,384):\n"
        "   Because the generator state space is deterministic and finite, once\n"
        "   X_{16384} is reached, the recurrence strictly repeats: X_{n + 16384} = X_n.\n"
        "   In a run of N = 100,000 draws, the exact same cycle of 16,384 numbers\n"
        "   repeats ~6.1035 times. No new values are ever produced; the stream\n"
        "   is composed of 16,384 unique values replicated 6 (or 7) times each.\n\n"
        "2. Why this does NOT manifest as a violation in a one-sample K-S test:\n"
        "   - The one-sample K-S test evaluates the 1-dimensional marginal cumulative\n"
        "     distribution function (ECDF) against the uniform CDF F(x) = x.\n"
        "   - Within one full cycle of 16,384 states, the generated values are\n"
        "     evenly dispersed across (0, 1).\n"
        "   - Repeating the entire cycle ~6.1 times replicates every value almost\n"
        "     equally, so the empirical CDF curve remains a smooth diagonal line\n"
        "     very close to y = x (D_N = 0.000464 << D_crit = 0.004301).\n"
        "   - Crucially, the K-S test is ORDER-INVARIANT (it operates solely on the\n"
        "     sorted sample U_(1) <= ... <= U_(N) and completely discards sequential\n"
        "     ordering / serial correlation).\n"
        "   - A sequence can be 100% deterministic, short-cycled, and severely\n"
        "     autocorrelated, yet still exhibit an ideal 1-D uniform marginal\n"
        "     distribution. To detect hidden period collapse, one MUST perform\n"
        "     independence tests (autocorrelation, runs test) or explicit cycle analysis."
    )
    print("=" * 70)


def plot_c2_investigation(show=True, save_path=None):
    """
    Plot C2 investigation diagnostics: ECDF vs Theoretical Uniform (showing why K-S passes)
    and Recurrence Scatter (showing the hidden period collapse p=16384).

    Parameters
    ----------
    show : bool, default=True
        Whether to call plt.show().
    save_path : str, optional
        If provided, save the figure to this file path.
    """
    import matplotlib.pyplot as plt

    sample = lcg_generator(100000)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. ECDF vs Uniform Diagonal
    sub = sorted(sample[::100])
    m_sub = len(sub)
    axes[0].plot([0, 1], [0, 1], "r--", linewidth=2, label="Theoretical Uniform CDF $F(x)=x$")
    axes[0].plot(sub, [(i + 1) / m_sub for i in range(m_sub)], "b-", linewidth=1.5, label="Empirical CDF $S_N(x)$ (N=100,000)")
    axes[0].set_title("K-S Test: Empirical CDF vs Theoretical\nPasses Uniformity ($D_N = 0.00046 < D_{crit} = 0.00430$)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Cumulative Probability")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper left")

    # 2. Recurrence Scatter Plot U_n vs U_{n + 16384}
    p = 16384
    axes[1].scatter(sample[:5000], sample[p:p + 5000], color="purple", s=6, alpha=0.6)
    axes[1].plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Exact Match Line $U_{n+p} = U_n$")
    axes[1].set_title("Hidden Period Collapse Check ($U_n$ vs $U_{n+16384}$)\nPerfect Diagonal Confirms $p = 16,384$", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("$U_n$")
    axes[1].set_ylabel("$U_{n + 16384}$")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    task1_generate()
    task2_ks_test()
    task3_period_and_reflection()
