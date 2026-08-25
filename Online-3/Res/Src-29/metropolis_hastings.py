"""
Metropolis-Hastings (MH) Algorithm

Purpose:
    Generate samples from a target distribution p(x) that is known only
    up to a normalizing constant (i.e. we can evaluate p(x) but don't
    need to know the constant that makes it integrate to 1). This is
    the typical situation in Bayesian inference and complex simulation
    models where the normalizing constant is intractable.

Target distribution used here (appropriate example: a bimodal mixture
of two Gaussians, which is hard to sample directly but easy to evaluate):

    p(x) ~ 0.3 * N(x; mu=0,  sigma=1)  +  0.7 * N(x; mu=10, sigma=2)

Proposal distribution:
    Symmetric random-walk proposal:  x' = x_current + N(0, step_size^2)
    Because the proposal is symmetric, q(x'|x) = q(x|x'), so the
    acceptance ratio simplifies to the plain Metropolis ratio:

        alpha = min( 1, p(x') / p(x_current) )

Algorithm:
    1. Start at an initial value x_0.
    2. At each iteration, propose x' from the proposal distribution.
    3. Compute acceptance probability alpha = min(1, p(x')/p(x_current)).
    4. Draw U ~ Uniform(0,1). If U <= alpha, accept x' (move there).
       Otherwise, reject and stay at x_current.
    5. Repeat for the desired number of iterations.
    6. Discard an initial "burn-in" period to let the chain converge.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# PARAMETERS (defined at the start)
# ---------------------------------------------------------------------
np.random.seed(42)          # reproducibility

n_iterations = 20000         # total number of MH iterations to run
burn_in      = 2000          # number of initial samples to discard
step_size    = 3.0           # std dev of the random-walk proposal N(0, step_size^2)
x0           = 0.0           # starting value of the chain

# ---------------------------------------------------------------------
# TARGET DISTRIBUTION p(x)  (unnormalized is fine -- MH only needs ratios)
# ---------------------------------------------------------------------
def target_pdf(x):
    def normal_pdf(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return 0.3 * normal_pdf(x, mu=0, sigma=1) + 0.7 * normal_pdf(x, mu=10, sigma=2)

# ---------------------------------------------------------------------
# METROPOLIS-HASTINGS SAMPLING LOOP
# ---------------------------------------------------------------------
samples = np.zeros(n_iterations)
x_current = x0
num_accepted = 0

for i in range(n_iterations):
    # 1. propose a new point via symmetric random walk
    x_proposed = x_current + np.random.normal(0, step_size)

    # 2. compute acceptance probability (Metropolis ratio, symmetric proposal)
    p_current = target_pdf(x_current)
    p_proposed = target_pdf(x_proposed)
    alpha = min(1.0, p_proposed / p_current) if p_current > 0 else 1.0

    # 3. accept or reject
    u = np.random.uniform(0, 1)
    if u <= alpha:
        x_current = x_proposed
        num_accepted += 1

    samples[i] = x_current

acceptance_rate = num_accepted / n_iterations

# ---------------------------------------------------------------------
# DISCARD BURN-IN PERIOD
# ---------------------------------------------------------------------
samples_post_burn_in = samples[burn_in:]

# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------
print(f"Total iterations       : {n_iterations}")
print(f"Burn-in discarded      : {burn_in}")
print(f"Samples retained       : {len(samples_post_burn_in)}")
print(f"Proposal step size     : {step_size}")
print(f"Acceptance rate        : {acceptance_rate:.4f}")
print(f"Sample mean            : {samples_post_burn_in.mean():.4f}")
print(f"Sample std dev         : {samples_post_burn_in.std():.4f}")
print("\nFirst 10 samples of the chain:")
print(np.round(samples[:10], 4))

# ---------------------------------------------------------------------
# PLOT: histogram of retained samples vs. the true target density
# ---------------------------------------------------------------------
x_grid = np.linspace(samples_post_burn_in.min() - 2, samples_post_burn_in.max() + 2, 1000)
true_density = target_pdf(x_grid)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# left panel: trace plot of the chain
axes[0].plot(samples, linewidth=0.5)
axes[0].axvline(burn_in, color="red", linestyle="--", label="end of burn-in")
axes[0].set_title("MH Chain Trace")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("x")
axes[0].legend()

# right panel: histogram of samples vs true target density
axes[1].hist(samples_post_burn_in, bins=80, density=True, alpha=0.6, label="MH samples (post burn-in)")
axes[1].plot(x_grid, true_density, color="black", linewidth=2, label="true target p(x)")
axes[1].set_title("Sampled Distribution vs. Target")
axes[1].set_xlabel("x")
axes[1].set_ylabel("density")
axes[1].legend()

plt.tight_layout()
plt.savefig("mh_algorithm_result.png", dpi=150)
print("\nPlot saved to mh_algorithm_result.png")