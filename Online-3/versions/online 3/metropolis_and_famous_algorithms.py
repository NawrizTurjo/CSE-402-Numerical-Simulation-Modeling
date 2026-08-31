"""Easy versions of Metropolis-Hastings and other common algorithms."""

import math


# ------------------------------------------------------------
# A simple random-number generator used by all examples
# ------------------------------------------------------------

seed = 12345


def random_number():
    """Return one pseudo-random value in [0, 1) using an LCG."""
    global seed
    seed = (1103515245 * seed + 12345) % (2**31)
    return seed / (2**31)


# ------------------------------------------------------------
# 1. METROPOLIS-HASTINGS ALGORITHM
# ------------------------------------------------------------

def target_density(x):
    """Density proportional to the standard normal distribution."""
    return math.exp(-(x * x) / 2)


def metropolis_hastings(start, total_samples, step_size):
    samples = []
    current = start
    accepted = 0

    for i in range(total_samples):
        # Propose a nearby value between current-step and current+step.
        proposal = current + (2 * random_number() - 1) * step_size

        # Proposal distribution is symmetric, so it cancels from the ratio.
        ratio = target_density(proposal) / target_density(current)
        acceptance_probability = min(1, ratio)

        if random_number() < acceptance_probability:
            current = proposal
            accepted += 1

        samples.append(current)

    acceptance_rate = accepted / total_samples
    return samples, acceptance_rate


samples, rate = metropolis_hastings(start=0, total_samples=10000, step_size=1)

# Ignore the first 1,000 samples (burn-in period).
useful_samples = samples[1000:]
sample_mean = sum(useful_samples) / len(useful_samples)

print("METROPOLIS-HASTINGS")
print("First 10 samples:", samples[:10])
print("Estimated mean:", round(sample_mean, 4))
print("Acceptance rate:", round(rate, 4))


# ------------------------------------------------------------
# 2. ACCEPTANCE-REJECTION METHOD
# Generate values with density f(x) = 2x on [0, 1].
# ------------------------------------------------------------

def acceptance_rejection(n):
    values = []

    while len(values) < n:
        x = random_number()
        y = 2 * random_number()

        # Accept the point if it is below f(x) = 2x.
        if y <= 2 * x:
            values.append(x)

    return values


rejection_values = acceptance_rejection(10)
print("\nACCEPTANCE-REJECTION")
print("Generated values:", rejection_values)


# ------------------------------------------------------------
# 3. XORSHIFT RANDOM-NUMBER GENERATOR
# Another famous and short pseudo-random generator.
# ------------------------------------------------------------

def xorshift(seed_value, n):
    values = []
    x = seed_value

    for i in range(n):
        x = x ^ ((x << 13) & 0xFFFFFFFF)
        x = x ^ (x >> 17)
        x = x ^ ((x << 5) & 0xFFFFFFFF)
        x = x & 0xFFFFFFFF
        values.append(x / (2**32))

    return values


print("\nXORSHIFT")
print("First 10 values:", xorshift(12345, 10))


# ------------------------------------------------------------
# 4. IMPORTANCE SAMPLING
# Estimate integral of x^2 from 0 to 1 using density p(x) = 2x.
# Its inverse CDF is x = sqrt(u).
# ------------------------------------------------------------

def importance_sampling(n):
    total = 0

    for i in range(n):
        u = random_number()
        x = math.sqrt(u)

        function_value = x * x
        sampling_density = 2 * x
        total += function_value / sampling_density

    return total / n


estimate = importance_sampling(10000)
print("\nIMPORTANCE SAMPLING")
print("Integral of x^2 from 0 to 1:", round(estimate, 4))
