"""
Practice 4 (Res/Src-26/practice-problems.md): 1D random walk.

100 steps, each +1 or -1 with equal probability. Estimate
P(|final position| > 15) over 50,000 walks. trial_fn simulates ONE
walk and returns the 0/1 indicator -- same probability-estimation
pattern as the Buffon's needle / project-completion problems.

Run standalone:
    python -m solutions.practice_src26.p4_random_walk
# """


# import sys
# from pathlib import Path

# # Allow direct script execution from any directory
# _CODE_DIR = Path(__file__).resolve().parent
# while _CODE_DIR.name != "Code" and _CODE_DIR.parent != _CODE_DIR:
#     _CODE_DIR = _CODE_DIR.parent
# if str(_CODE_DIR) not in sys.path:
#     sys.path.insert(0, str(_CODE_DIR))

# import random

# from monte_carlo.core import monte_carlo_estimate


# def random_walk_indicator(num_steps=100, threshold=15):
#     position = sum(random.choice((-1, 1)) for _ in range(num_steps))
#     return 1.0 if abs(position) > threshold else 0.0


# if __name__ == "__main__":
#     result = monte_carlo_estimate(random_walk_indicator, n=50000, seed=1)
#     print(f"P(|final position| > 15) = {result['estimate']:.4f}  "
#           f"CI95={tuple(round(v, 4) for v in result['ci95'])}")


import random
import math

def random_walk_simulation(n=100000, num_steps=100, threshold=15, seed=1):

    random.seed(seed)

    success = 0

    # Run the random walk many times
    for i in range(n):

        position = 0

        # One random walk = 100 steps
        for j in range(num_steps):
            step = (1 if random.random() < 0.5 else -1)
            position += step

        # Check if final position is farther than threshold
        if abs(position) > threshold:
            success += 1

    # Probability = successful trials / total trials
    probability = success / n

    # Standard error for probability
    std_error = math.sqrt(probability * (1 - probability) / n)

    # 95% confidence interval
    low = probability - 1.96 * std_error
    high = probability + 1.96 * std_error

    print("Estimated probability =", round(probability, 5))
    print("Standard error =", round(std_error, 5))
    print("95% CI =", (round(low, 5), round(high, 5)))


random_walk_simulation()