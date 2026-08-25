import math

def runs_test(R):
    """
    A 'run' = maximal unbroken sequence of values all above
    OR all below the mean (0.5).

    Too FEW runs → long streaks → positive autocorrelation
    Too MANY runs → alternating → negative autocorrelation
    """
    n = len(R)
    # Label: 1 if above 0.5, 0 if below
    above = [1 if r > 0.5 else 0 for r in R]
    # Count runs (every time the label changes, a new run starts)
    runs = 1
    for k in range(1, n):
        if above[k] != above[k-1]:
            runs += 1
    n1 = sum(above)       # count of values above 0.5
    n2 = n - n1           # count of values below 0.5
    # Expected runs and std dev under H0 (independence)
    E_runs = (2 * n1 * n2) / n + 1
    var_runs = (2*n1*n2*(2*n1*n2 - n)) / (n**2 * (n-1))
    Z0 = (runs - E_runs) / math.sqrt(var_runs)
    decision = "Reject H0 (DEPENDENT)" if abs(Z0) > 1.96 else "Fail to Reject H0"
    return runs, E_runs, Z0, decision


def main():
    sample = [
        0.23, 0.28, 0.33, 0.27, 0.05,
        0.36, 0.72, 0.81, 0.44, 0.91,
        0.12, 0.67, 0.53, 0.39, 0.88,
    ]

    runs, expected_runs, z0, decision = runs_test(sample)

    print("Runs Test for Independence")
    print("Sample values:", sample)
    print(f"Number of runs: {runs}")
    print(f"Expected runs: {expected_runs:.4f}")
    print(f"Z0: {z0:.4f}")
    print(f"Decision: {decision}")


if __name__ == "__main__":
    main()

