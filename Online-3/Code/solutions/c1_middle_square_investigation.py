"""
C1 (online exam, section C1, see Res/questions.txt): Investigating the
Middle-Square Random Number Generator.

All four tasks are answered here purely by calling the reusable
rng/middle_square.py and testing/chi_square_test.py modules -- this file
just wires them together and prints the required tables.

Run standalone:
    python -m solutions.c1_middle_square_investigation
"""


import sys
from pathlib import Path

# Allow direct script execution from any directory
_CODE_DIR = Path(__file__).resolve().parent
while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
    _CODE_DIR = _CODE_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from rng.middle_square import middle_square, middle_square_uniforms, find_middle_square_cycle
from testing.chi_square_test import chi_square_uniform_test

PROBLEMATIC_SEED = 2500  # 2500^2 = 06250000 -> middle four digits = 2500 (stuck immediately)


def task1_generate():
    seed = 5731
    values = middle_square(seed, 100)
    print(f"TASK 1: 100 values generated from seed {seed}")
    print(values)
    print()
    return values


def task2_edge_case():
    repeated_value, repeat_iteration, cycle_length = find_middle_square_cycle(PROBLEMATIC_SEED)
    print("TASK 2: Edge case")
    print(f"  Seed:                     {PROBLEMATIC_SEED}")
    print(f"  First repeated value:     {repeated_value}")
    print(f"  Iteration at which it repeats: {repeat_iteration}")
    print(f"  Cycle length:             {cycle_length}")
    print()


def task3_chi_square():
    print("TASK 3: Chi-Square test, N=1000, 10 bins, alpha=0.05")
    print("| Seed             | Chi^2    | p-value   | Decision          |")
    print("|------------------|----------|-----------|-------------------|")
    for label, seed in [("5731", 5731), ("6239", 6239), ("Problematic Seed", PROBLEMATIC_SEED)]:
        uniforms = middle_square_uniforms(seed, 1000)
        result = chi_square_uniform_test(uniforms)
        print(f"| {label:<16} | {result['chi2']:8.4f} | {result['p_value']:.6g} | {result['decision']:<17} |")
    print()


def task4_reflect():
    print("TASK 4: Reflection")
    print(
        "No. Passing the Chi-Square test only shows the ONE-DIMENSIONAL bin\n"
        "counts look uniform in aggregate -- it says nothing about how\n"
        "consecutive values relate to each other, or how short the generator's\n"
        "cycle really is. This dataset makes that concrete: seeds 5731 and 6239\n"
        "look like 'normal' seeds (no obvious problem in the first 10 values),\n"
        "yet find_middle_square_cycle shows BOTH collapse into the same short\n"
        "4-value cycle (6100 -> 2100 -> 4100 -> 8100 -> 6100 -> ...) well before\n"
        "1000 iterations (at iteration 75 and 111 respectively). Once a stream\n"
        "is dominated by a repeating cycle, its Chi-Square statistic explodes\n"
        "and it correctly REJECTS H0 -- so here Chi-Square actually catches the\n"
        "problem. But it would NOT catch every problem: a generator can have\n"
        "strong autocorrelation/lattice structure that is invisible to a 1-D\n"
        "bin count while still passing Chi-Square (see rng/lcg.py's RANDU note:\n"
        "passes 1-D uniformity, fails badly once you look at consecutive\n"
        "TRIPLES). A trustworthy RNG needs to pass uniformity tests (Chi-Square,\n"
        "K-S) AND independence tests (autocorrelation, runs) AND have an\n"
        "adequately long period for the intended use -- Chi-Square alone checks\n"
        "only the first of these."
    )


def plot_c1_investigation(show=True, save_path=None):
    """
    Plot sequence trajectories and bin frequency comparisons for seeds 5731, 6239, and 2500.

    Parameters
    ----------
    show : bool, default=True
        Whether to call plt.show().
    save_path : str, optional
        If provided, save the figure to this file path.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Trajectories showing cycle collapse
    seeds = [(5731, "Seed 5731", "royalblue"), (6239, "Seed 6239", "forestgreen"), (2500, "Seed 2500 (Degenerate)", "crimson")]
    for s, label, col in seeds:
        vals = middle_square_uniforms(s, 150)
        axes[0].plot(vals, label=label, color=col, alpha=0.8, linewidth=1.5)

    axes[0].set_title("Sequence Trajectories (First 150 Iterations)\nShows Rapid Cycle Collapse", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("$U_n$")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # 2. Chi-Square Bin Frequencies for Seed 5731 (N=1000)
    u_5731 = middle_square_uniforms(5731, 1000)
    res = chi_square_uniform_test(u_5731, bins=10)
    bin_labels = [f"B{i+1}" for i in range(10)]
    axes[1].bar(bin_labels, res["counts"], color="coral", edgecolor="black", alpha=0.8, label="Observed $O_i$")
    axes[1].axhline(100.0, color="black", linestyle="--", linewidth=2, label="Expected $E_i = 100$")
    axes[1].set_title(f"Seed 5731 Bin Frequencies (N=1000)\n$\\chi^2 = {res['chi2']:.1f}$ (Explodes due to 4-value cycle)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Interval Bins")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    task1_generate()
    task2_edge_case()
    task3_chi_square()
    task4_reflect()
