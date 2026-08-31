import math
import random

def monte_carlo_estimate(trial_fn, n, seed=None):
    """
    Run `trial_fn()` n times (each call = one independent random trial
    returning a single float) and return the sample-mean estimate with
    its standard error and a 95% confidence interval.

    Parameters
    ----------
    trial_fn : callable
        Zero-argument function. Each call performs ONE random trial and
        returns a numeric outcome (0/1 for probabilities, a payoff for
        expected value, box_volume*indicator for hit-or-miss volume,
        width*f(x) for integration -- see module docstring above).
    n : number of independent trials to run.
    seed : optional seed passed to `random.seed()` for reproducibility.
           trial_fn must use the shared `random` module (not its own
           generator) for this to have any effect.

    Returns
    -------
    dict with:
        estimate   : sample mean = the Monte Carlo answer.
        std_dev    : sample standard deviation of the n outcomes.
        std_error  : std_dev / sqrt(n)  (this is why error shrinks as 1/sqrt(n),
                     REGARDLESS of dimension -- the key advantage of MC
                     over grid methods like Simpson's rule in high-D).
        ci95       : (low, high) 95% confidence interval, mean +/- 1.96*std_error.
    """
    if seed is not None:
        random.seed(seed)

    total = 0.0
    total_sq = 0.0
    for _ in range(n):
        y = trial_fn()
        total += y
        total_sq += y * y

    mean = total / n
    variance = max(0.0, (total_sq - n * mean * mean) / (n - 1)) if n > 1 else 0.0
    std_dev = math.sqrt(variance)
    std_error = std_dev / math.sqrt(n)

    return {
        "estimate": mean,
        "std_dev": std_dev,
        "std_error": std_error,
        "ci95": (mean - 1.96 * std_error, mean + 1.96 * std_error),
    }

def integrate_1d(f, a, b, n, seed=None):
    """
    Estimate integral_a^b f(x) dx via the sample-mean method:
        X ~ Uniform(a, b),  Y = (b - a) * f(X),  estimate = mean(Y).
    Works for ANY f, even ones with no closed-form antiderivative
    (e.g. sin(x)*cos(x^2)) -- that's the whole point of Monte Carlo
    integration.
    """
    width = b - a

    def trial():
        x = random.uniform(a, b)
        return width * f(x)

    return monte_carlo_estimate(trial, n, seed=seed)


def estimate_pi_hit_or_miss(n, seed=None):
    """
    Classic pi estimation: throw darts at the unit square [0,1]x[0,1],
    check whether they land inside the quarter unit circle (x^2+y^2<=1).
    P(inside) = (pi/4 * 1^2) / (1*1) = pi/4   =>   pi = 4 * P(inside).
    This is a hit-or-miss volume estimate in disguise: box area = 1,
    trial returns 4 * indicator (the "4" folds the box-area scaling in).
    """
    def trial():
        x, y = random.random(), random.random()
        return 4.0 if x * x + y * y <= 1.0 else 0.0

    return monte_carlo_estimate(trial, n, seed=seed)


def metropolis_hastings(
    target_density,
    x0,
    n_samples,
    proposal_std=1.0,
    seed=None
):
    
    if seed is not None:
        random.seed(seed)
    x_current = x0

    samples = [x_current]

    accepted = 0

    for _ in range(n_samples):

        epsilon = random.gauss(0, proposal_std)

        x_proposed = x_current + epsilon

        current_probability = target_density(x_current)

        proposed_probability = target_density(x_proposed)

        acceptance_ratio = (
            proposed_probability / current_probability
        )

        acceptance_probability = min(
            1.0,
            acceptance_ratio
        )

        random_number = random.random()

        if random_number < acceptance_probability:
            x_current = x_proposed
            accepted += 1
        else:
            x_current = x_current

        samples.append(x_current)
    acceptance_rate = accepted / n_samples

    return samples, acceptance_rate

def normal_density(x, mean=3.0, std=0.5):
    """
    Probability density of a normal distribution.
    """

    coefficient = 1 / (
        math.sqrt(2 * math.pi) * std
    )

    exponent = -((x - mean) ** 2) / (
        2 * std ** 2
    )

    probability = coefficient * math.exp(exponent)

    return probability

samples, acceptance_rate = metropolis_hastings(
    target_density=normal_density,
    x0=0.0,
    n_samples=5000,
    proposal_std=1.0,
    seed=42
)

burn_in = 500

kept_samples = samples[burn_in:]

def calculate_mean(values):
    total = 0

    for value in values:
        total += value

    mean = total / len(values)

    return mean


def calculate_variance(values):
    mean = calculate_mean(values)

    total = 0

    for value in values:
        difference = value - mean
        total += difference ** 2

    variance = total / len(values)

    return variance

mean = calculate_mean(kept_samples)
variance = calculate_variance(kept_samples)

print("Acceptance rate :", acceptance_rate)
print("Sample mean     :", mean)
print("Sample variance :", variance)

print("True mean       :", 3.0)
print("True variance   :", 0.25)