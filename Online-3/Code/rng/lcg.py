"""
Linear Congruential Generator (LCG):

    X_{i+1} = (a * X_i + c) mod m         R_i = X_i / m   (in [0, 1))

- c != 0  -> "mixed" LCG
- c == 0  -> "multiplicative" LCG (a.k.a. Lehmer generator)

Maximum-period theory (Hull-Dobell + related results, check these by hand
or with `check_max_period_conditions` below):

    Case 1 -- m = 2^b, c != 0 (mixed):
        Full period m is achieved  iff  gcd(c, m) == 1  AND  (a - 1) % 4 == 0.

    Case 2 -- m = 2^b, c == 0 (multiplicative):
        Max possible period is m/4 (never m, because 0 is an absorbing/
        excluded state and even seeds fall into a shorter sub-cycle).
        Achieved iff  seed is ODD  AND  (a % 8 == 3  or  a % 8 == 5).

    Case 3 -- m prime, c == 0 (multiplicative):
        Max period m - 1 (every nonzero residue) iff `a` is a PRIMITIVE
        ROOT modulo m, i.e. the multiplicative order of a mod m equals m-1.

Worked example (slide's mixed LCG): seed=27, a=17, c=43, m=100
    X1 = (17*27 + 43) % 100 = 502 % 100 = 2    -> R1 = 0.02
    X2 = (17*2 + 43)  % 100 = 77              -> R2 = 0.77
    X3 = (17*77 + 43) % 100 = 1352 % 100 = 52  -> R3 = 0.52
    X4 = (17*52 + 43) % 100 = 927 % 100 = 27   -> R4 = 0.27  (back to seed! period 4)

Known weakness worth remembering for a "reflect" question: IBM's RANDU
(a=65539, c=0, m=2^31) passes 1-D uniformity tests but consecutive
TRIPLES (X_i, X_{i+1}, X_{i+2}) all satisfy
    9*X_i - 6*X_{i+1} + X_{i+2} = 0  (mod m)
so every triple lies on just 15 parallel 2-D planes instead of filling
3-D space -- passing a 1-D Chi-Square test does NOT guarantee a generator
is good in higher dimensions or against other tests (independence, etc).

Run standalone:
    python -m rng.lcg
"""


import sys
from pathlib import Path

# Allow direct script execution from any directory
_CODE_DIR = Path(__file__).resolve().parent
while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
    _CODE_DIR = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from math import gcd

from rng.common import find_cycle, bin_counts


def lcg_step(x, a, c, m):
    return (a * x + c) % m


def lcg(seed, a, c, m, n):
    """Generate n successive integer states X1..Xn."""
    values = []
    x = seed
    for _ in range(n):
        x = lcg_step(x, a, c, m)
        values.append(x)
    return values


def lcg_uniforms(seed, a, c, m, n):
    """Same sequence as lcg(), rescaled to floats in [0, 1)."""
    return [x / m for x in lcg(seed, a, c, m, n)]


def find_lcg_cycle(seed, a, c, m):
    """Locate the first repeat, its iteration, and the cycle length (=period from this seed)."""
    return find_cycle(lambda x: lcg_step(x, a, c, m), seed)


def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def check_max_period_conditions(a, c, m, seed):
    """
    Checks which theoretical max-period case applies and whether its
    conditions are satisfied. Returns (case_description, theoretical_max_period,
    conditions_satisfied, details_string). Falls back to "measure it yourself
    with find_lcg_cycle" when m doesn't match a known closed-form case
    (e.g. m composite and not a power of 2).
    """
    is_power_of_two = m > 0 and (m & (m - 1)) == 0

    if is_power_of_two and c != 0:
        cond1 = gcd(c, m) == 1
        cond2 = (a - 1) % 4 == 0
        return ("Case 1: m=2^b, mixed (c!=0)", m, cond1 and cond2,
                f"gcd(c,m)==1: {cond1}, (a-1)%4==0: {cond2}")

    if is_power_of_two and c == 0:
        cond1 = seed % 2 == 1
        cond2 = a % 8 in (3, 5)
        return ("Case 2: m=2^b, multiplicative (c=0)", m // 4, cond1 and cond2,
                f"seed is odd: {cond1}, a%8 in {{3,5}}: {cond2}")

    if c == 0 and is_prime(m):
        # a is a primitive root mod m  <=>  order of a divides (m-1) and equals it exactly.
        order, val = 1, a % m
        while val != 1 and order < m:
            val = (val * a) % m
            order += 1
        return ("Case 3: m prime, multiplicative (c=0)", m - 1, order == m - 1,
                f"multiplicative order of a mod m = {order}")

    return ("General case (no closed form here)", None, None,
            "use find_lcg_cycle(seed, a, c, m) to measure the period directly")


def plot_lcg(uniforms, title="LCG Output Diagnostics", show=True, save_path=None):
    """
    Plot LCG output diagnostics: sequence trajectory, 1-D histogram, and 2-D lag-1 scatter plot.

    Parameters
    ----------
    uniforms : list of float
        Output variates in [0, 1).
    title : str, default='LCG Output Diagnostics'
        Figure title.
    show : bool, default=True
        Whether to call plt.show().
    save_path : str, optional
        If provided, save the figure to this file path.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Sequence trace
    axes[0].plot(uniforms[:200], marker="o", markersize=3, linewidth=1, color="royalblue")
    axes[0].set_title("Sequence (First 200 values)")
    axes[0].set_xlabel("Iteration (i)")
    axes[0].set_ylabel("R_i")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Histogram
    axes[1].hist(uniforms, bins=10, range=(0, 1), density=True, color="mediumpurple", edgecolor="black", alpha=0.7)
    axes[1].axhline(1.0, color="red", linestyle="--", label="Ideal Uniform(0,1)")
    axes[1].set_title(f"Histogram (N={len(uniforms)})")
    axes[1].set_xlabel("R_i")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Lag-1 Scatter Plot (R_i vs R_{i+1})
    if len(uniforms) > 1:
        axes[2].scatter(uniforms[:-1], uniforms[1:], s=10, color="forestgreen", alpha=0.5)
    axes[2].set_title("Lag-1 Scatter: $R_i$ vs $R_{i+1}$")
    axes[2].set_xlabel("$R_i$")
    axes[2].set_ylabel("$R_{i+1}$")
    axes[2].set_xlim(-0.05, 1.05)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    print("Mixed LCG example: seed=27, a=17, c=43, m=100")
    values = lcg(27, 17, 43, 100, n=4)
    print("X1..X4:", values)

    repeated, iteration, period = find_lcg_cycle(27, 17, 43, 100)
    print(f"Period from this seed: {period}")

    print("\nMultiplicative LCG: X_{i+1} = 13*X_i mod 64, checking seeds 1-4")
    for s in (1, 2, 3, 4):
        _, _, period = find_lcg_cycle(s, a=13, c=0, m=64)
        case, max_p, ok, details = check_max_period_conditions(a=13, c=0, m=64, seed=s)
        print(f"  seed={s}: measured period={period:2d}  |  {case} -> max={max_p}, satisfied={ok}")

    uniforms = lcg_uniforms(27, 17, 43, 100, n=1000)
    print(f"\nBin counts (10 bins) for {len(uniforms)} values: {bin_counts(uniforms)}")
