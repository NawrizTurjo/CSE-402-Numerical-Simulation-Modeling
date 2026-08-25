"""
CSE 401: Middle-Square Random Number Generator (Simple Implementation)
Tasks 1 to 4 from Section C1 Online-3.
"""

from scipy.stats import chisquare

def middle_square(seed, n):
    results = []
    x = seed
    for _ in range(n):
        square = x * x
        sq_str = str(square).zfill(8)
        middle_str = sq_str[2:6]
        x = int(middle_str)
        results.append(x / 10000.0)
    return results

def find_cycle(seed):
    seen = {}
    x = seed
    iteration = 0
    while x not in seen:
        seen[x] = iteration
        sq_str = str(x * x).zfill(8)
        x = int(sq_str[2:6])
        iteration += 1
    return x, iteration, iteration - seen[x]

def perform_chi_square_test(numbers, k=10, alpha=0.05):
    observed = [0] * k
    for r in numbers:
        bin_idx = min(int(r * k), k - 1)
        observed[bin_idx] += 1
    chi2_stat, p_value = chisquare(observed)
    decision = "Reject H0" if p_value < alpha else "Do not reject H0"
    return chi2_stat, p_value, decision, observed

if __name__ == "__main__":
    print("TASK 1: Generate 100 values from seed 5731:")
    sample = middle_square(5731, 100)
    print("First 10 values:", [round(v, 4) for v in sample[:10]])

    print("\nTASK 2: Edge Case with Seed 3792:")
    val, it, cycle = find_cycle(3792)
    print(f"Seed: 3792 | First Repeated: {val} | Iteration: {it} | Cycle: {cycle}")

    print("\nTASK 3: Chi-Square Test (N = 1000):")
    for s in (5731, 6239, 3792):
        chi2, p_val, dec, _ = perform_chi_square_test(middle_square(s, 1000))
        print(f"Seed {s:4d} -> Chi^2: {chi2:8.2f} | p-value: {p_val:10.4e} | Decision: {dec}")
