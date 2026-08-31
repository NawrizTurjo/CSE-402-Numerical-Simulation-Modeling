import numpy as np
import random
import matplotlib.pyplot as plt

def target_distribution(x):

    return np.exp(-0.5 * x**2)


def metropolis_hastings(
    iterations=10000,
    initial=0,
    proposal_std=1
):

    samples = []

    current = initial

    for _ in range(iterations):

        # Generate proposal
        proposal = np.random.normal(
            current,
            proposal_std
        )

        # Acceptance ratio
        current_prob = target_distribution(current)
        proposal_prob = target_distribution(proposal)

        acceptance_ratio = min(
            1,
            proposal_prob / current_prob
        )

        # Accept or reject
        if random.random() < acceptance_ratio:
            current = proposal

        samples.append(current)

    return samples


samples = metropolis_hastings()

print("Mean:", np.mean(samples))
print("Std:", np.std(samples))

plt.hist(
    samples,
    bins=50,
    density=True
)

x = np.linspace(-4, 4, 200)

normal_pdf = (
    1 / np.sqrt(2 * np.pi)
) * np.exp(-x*x / 2)

plt.plot(
    x,
    normal_pdf,
    label="True Normal PDF"
)

plt.xlabel("x")
plt.ylabel("Density")
plt.title("Metropolis-Hastings Sampling")

plt.legend()
plt.show()