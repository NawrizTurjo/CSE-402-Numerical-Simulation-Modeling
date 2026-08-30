"""
Metropolis-Hastings (MH) algorithm -- MCMC sampling from a target
distribution p(x) that you can only EVALUATE (even just up to an unknown
normalizing constant), not sample from directly.

Algorithm (symmetric random-walk proposal, the version covered in the
syllabus):
    1. Start at x0.
    2. Propose x' = x_current + Normal(0, proposal_std^2).
       (Symmetric proposal: q(x'|x) = q(x|x'), so it cancels out of the
        acceptance ratio below -- this simplified form is "Metropolis",
        the general asymmetric-proposal form is "Metropolis-Hastings".)
    3. Acceptance ratio: alpha = min(1, p(x') / p(x_current)).
       In log form (safer -- avoids overflow/underflow for extreme densities):
           log_alpha = log_target(x') - log_target(x_current)
           accept if  log(U) < log_alpha,  U ~ Uniform(0,1)
    4. If accepted, move to x'. Otherwise stay at x_current (this repeat
       is still recorded as a sample -- that's what makes it a Markov chain).
    5. Repeat for n_samples iterations, then discard an initial "burn-in"
       prefix so the chain has time to reach the target distribution
       before you start trusting its samples.

Why it works (intuition, not proof): the acceptance rule is built so
that, in the long run, the chain visits each state x with frequency
proportional to p(x) -- this is the "detailed balance" property. You
don't need to know the normalizing constant of p(x) because it cancels
in the ratio p(x')/p(x_current).

Run standalone:
    python -m monte_carlo.metropolis_hastings
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
import random


def metropolis_hastings(log_target, x0, n_samples, proposal_std=1.0, seed=None):
    """
    log_target   : function returning log of the UNNORMALIZED target density.
    x0           : starting state.
    n_samples    : number of iterations to run (chain length, excluding x0).
    proposal_std : std dev of the symmetric Gaussian proposal (step size);
                   too small -> slow exploration, too large -> most
                   proposals rejected. Tune so acceptance rate ~20-50%.

    Returns (samples, acceptance_rate). samples[0] == x0.
    """
    if seed is not None:
        random.seed(seed)

    x = x0
    samples = [x]
    accepted = 0

    for _ in range(n_samples):
        x_proposed = x + random.gauss(0, proposal_std)
        log_alpha = log_target(x_proposed) - log_target(x)

        if math.log(random.random()) < log_alpha:
            x = x_proposed
            accepted += 1

        samples.append(x)

    return samples, accepted / n_samples


def mean_and_variance(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, var


def plot_metropolis_hastings(samples, target_pdf=None, burn_in=500, title="Metropolis-Hastings Diagnostics", show=True, save_path=None):
    """
    Plot Markov Chain trace and histogram of retained samples against true target density.

    Parameters
    ----------
    samples : list of float
        All generated samples.
    target_pdf : callable, optional
        Target probability density function.
    burn_in : int, default=500
        Number of initial samples discarded.
    title : str, default='Metropolis-Hastings Diagnostics'
        Figure title.
    show : bool, default=True
        Whether to call plt.show().
    save_path : str, optional
        If provided, save the figure to this file path.
    """
    import matplotlib.pyplot as plt

    kept = samples[burn_in:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left panel: trace plot
    axes[0].plot(samples, linewidth=0.6, color="royalblue")
    axes[0].axvline(burn_in, color="crimson", linestyle="--", linewidth=1.5, label=f"End of Burn-in ({burn_in})")
    axes[0].set_title("MCMC Chain Trace")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("x")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Right panel: histogram vs target density
    axes[1].hist(kept, bins=50, density=True, color="mediumpurple", edgecolor="black", alpha=0.7, label="Retained Samples")
    if target_pdf is not None:
        min_x, max_x = min(kept), max(kept)
        grid = [min_x + (max_x - min_x) * i / 300.0 for i in range(301)]
        pdf_vals = [target_pdf(x) for x in grid]
        axes[1].plot(grid, pdf_vals, "k-", linewidth=2, label="Target PDF $p(x)$")
    axes[1].set_title("Sampled Density vs Target")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    # Sample from N(mu=3, sigma=0.5): log p(x) = -(x-3)^2 / (2 * 0.5^2), constant dropped.
    log_target = lambda x: -((x - 3) ** 2) / (2 * 0.5**2)

    samples, acceptance_rate = metropolis_hastings(
        log_target, x0=0.0, n_samples=5000, proposal_std=1.0, seed=42
    )

    burn_in = 500
    kept = samples[burn_in:]
    mean, variance = mean_and_variance(kept)

    print(f"Acceptance rate : {acceptance_rate:.4f}")
    print(f"Sample mean     : {mean:.4f}  (true = 3.0)")
    print(f"Sample variance : {variance:.4f}  (true = 0.25)")
