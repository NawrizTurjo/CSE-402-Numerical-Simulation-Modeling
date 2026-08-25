"""
Practice MH1: Sampling a Gamma(3,1)-shaped density with Metropolis-Hastings
(see ../../../Practice/PRACTICE_QUESTIONS.md).

Target (unnormalized): p(x) proportional to x^2 * e^(-x) for x > 0, and
0 for x <= 0. This is the shape of a Gamma(shape=3, rate=1) distribution
(true mean = shape/rate = 3, true variance = shape/rate^2 = 3), which is
easy to check your sampler against without needing scipy.

Handling the x <= 0 boundary: log_target returns -inf there, so
log_alpha = log_target(x') - log_target(x) is -inf whenever a proposal
lands at or below 0 -- meaning math.log(random.random()) < log_alpha is
never true, so the chain naturally never accepts a move into the
disallowed region. No special-case branch needed in the MH loop itself.

Run standalone:
    python -m solutions.practice_generated.mh1_gamma_sampling
"""

import math

from monte_carlo.metropolis_hastings import metropolis_hastings, mean_and_variance


def log_target(x):
    if x <= 0:
        return float("-inf")
    return 2 * math.log(x) - x  # log(x^2 * e^-x), constant normalizer dropped


def task1_log_target():
    print("TASK 1: log_target(x) = 2*log(x) - x for x>0, else -inf")
    for x in (0.5, 1.0, 3.0, -1.0):
        print(f"  log_target({x}) = {log_target(x)}")
    print()


def task2_sample_and_check():
    print("TASK 2: sample and check against the known Gamma(3,1) mean/variance")
    samples, acceptance_rate = metropolis_hastings(
        log_target, x0=1.0, n_samples=5000, proposal_std=1.5, seed=42
    )
    kept = samples[500:]  # discard burn-in
    mean, variance = mean_and_variance(kept)
    print(f"  proposal_std=1.5  acceptance_rate={acceptance_rate:.4f}")
    print(f"  sample mean     = {mean:.4f}  (true = 3.0)")
    print(f"  sample variance = {variance:.4f}  (true = 3.0)")
    print()


def task3_step_size_effect():
    print("TASK 3: effect of the proposal step size on acceptance rate")
    for step in (0.5, 1.5, 5.0):
        samples, rate = metropolis_hastings(log_target, x0=1.0, n_samples=5000, proposal_std=step, seed=42)
        mean, variance = mean_and_variance(samples[500:])
        print(f"  proposal_std={step:4.1f}  acceptance={rate:.4f}  mean={mean:.4f}  variance={variance:.4f}")
    print("  Small steps -> most proposals accepted but the chain explores slowly.")
    print("  Large steps -> big jumps proposed but most land in low-density area")
    print("  and get rejected, so acceptance rate drops. A moderate step size")
    print("  (here ~1.5) balances exploration against acceptance rate.")


if __name__ == "__main__":
    task1_log_target()
    task2_sample_and_check()
    task3_step_size_effect()
