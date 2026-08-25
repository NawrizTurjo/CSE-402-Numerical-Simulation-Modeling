import math, random

def metropolis_hastings(log_target, x0, n_samples, proposal_std=1.0, seed=None):
    """
    Sample from a distribution proportional to exp(log_target(x)).
    Uses symmetric Gaussian proposal.

    log_target   : function returning log of UNNORMALIZED target density
    x0           : starting state (any reasonable value)
    n_samples    : how many samples to draw
    proposal_std : σ of the Gaussian proposal — controls step size
    """
    if seed is not None:
        random.seed(seed)

    x = x0
    samples = [x]
    accepted = 0

    for _ in range(n_samples):
        # Step 1: propose new state
        x_proposed = x + random.gauss(0, proposal_std)

        # Step 2: log acceptance ratio (normalizing constant cancels out)
        log_alpha = log_target(x_proposed) - log_target(x)

        # Step 3: accept or reject
        if math.log(random.random()) < log_alpha:
            x = x_proposed    # ACCEPT — move to proposed state
            accepted += 1
        # else REJECT — stay at current x (x is unchanged)

        samples.append(x)

    return samples, accepted / n_samples  # return samples and acceptance rate


# Example 1: Sample from N(0,1)
# π(x) = exp(-x²/2) / Z
# log π(x) = -x²/2  (drop Z — it's a constant)
log_normal = lambda x: -x**2 / 2

samples, rate = metropolis_hastings(log_normal, x0=0.0, n_samples=5000, seed=42)
# acceptance rate ≈ 70%
# sample mean     ≈ 0.0   (true = 0)
# sample variance ≈ 1.0   (true = 1)


# Example 2: Sample from N(μ=3, σ=0.5)
# log π(x) = -(x-3)² / (2 × 0.5²)
log_shifted = lambda x: -(x-3)**2 / (2 * 0.5**2)

samples, rate = metropolis_hastings(log_shifted, x0=0.0, n_samples=5000, seed=42)
# sample mean     ≈ 3.0   (true = 3)
# sample variance ≈ 0.25  (true = 0.25)


# Example 3: Bimodal (two peaks at -2 and +2)
