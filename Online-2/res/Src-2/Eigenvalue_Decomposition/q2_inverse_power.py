import numpy as np

def inverse_power_iteration(A, tol=1e-9, max_iter=1000):
    """
    Finds the smallest (in magnitude) eigenvalue of A using the
    Inverse Power Method.
    Instead of computing A^{-1} directly, it solves A * y = x at each step.
    """
    n = A.shape[0]
    x = np.ones(n)
    x = x / np.linalg.norm(x)

    eigenvalue = 0.0
    print("\n--- Inverse Power Iteration ---")
    for iteration in range(1, max_iter + 1):
        # Solve A * y = x (instead of computing A^{-1} * x)
        y = np.linalg.solve(A, x)

        # The largest-magnitude entry of y is 1/lambda_min
        mu = y[np.argmax(np.abs(y))]

        # Normalize
        x_new = y / mu

        # The smallest eigenvalue estimate
        eigenvalue_new = 1.0 / mu

        if np.abs(eigenvalue_new - eigenvalue) < tol:
            print(f"Converged after {iteration} iterations.")
            eigenvalue = eigenvalue_new
            x = x_new
            break

        eigenvalue = eigenvalue_new
        x = x_new

    eigenvector = x / np.linalg.norm(x)
    return eigenvalue, eigenvector


if __name__ == "__main__":
    A = np.array([
        [4, 1, 0, 0],
        [1, 5, 1, 0],
        [0, 1, 6, 1],
        [0, 0, 1, 7]
    ], dtype=float)

    print("Matrix A:")
    print(A)

    smallest_eval, smallest_evec = inverse_power_iteration(A)

    print(f"\nSmallest Eigenvalue  : {smallest_eval:.6f}")
    print(f"Eigenvector (normalized): {np.round(smallest_evec, 6)}")

    # NumPy Verification
    evals, evecs = np.linalg.eig(A)
    min_idx = np.argmin(np.abs(evals))
    print(f"\nNumPy Smallest Eigenvalue : {evals[min_idx].real:.6f}")
    print(f"NumPy Eigenvector         : {np.round(evecs[:, min_idx].real, 6)}")
