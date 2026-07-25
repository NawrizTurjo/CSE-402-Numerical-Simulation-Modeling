import numpy as np

def power_iteration(A, tol=1e-9, max_iter=1000):
    """Standard Power Iteration — returns dominant eigenvalue and normalized eigenvector."""
    n = A.shape[0]
    x = np.ones(n)
    eigenvalue = 0.0

    for _ in range(max_iter):
        y = np.dot(A, x)
        eigenvalue_new = y[np.argmax(np.abs(y))]
        x = y / eigenvalue_new
        if np.abs(eigenvalue_new - eigenvalue) < tol:
            eigenvalue = eigenvalue_new
            break
        eigenvalue = eigenvalue_new

    eigenvector = x / np.linalg.norm(x)
    return eigenvalue, eigenvector


def hotellings_deflation(A, lam1, v1):
    """
    Performs Hotelling's Deflation to remove the contribution of the
    dominant eigenvalue from A, enabling the Power Method to find
    the second-largest eigenvalue.

    A2 = A - lambda1 * v1_hat * v1_hat^T
    """
    # Normalize eigenvector to unit length
    v1_hat = v1 / np.linalg.norm(v1)

    # lambda1 * v1 * v1T represents the contribution of first eigenpair
    # Subtract the dominant eigenvalue's contribution
    A2 = A - lam1 * np.outer(v1_hat, v1_hat)

    return A2


if __name__ == "__main__":
    # Use a symmetric matrix (deflation works cleanly for symmetric matrices)
    A = np.array([
        [4, 1, 0],
        [1, 3, 1],
        [0, 1, 2]
    ], dtype=float)

    print("Original Matrix A:")
    print(A)

    # Step 1: Find the dominant eigenvalue
    lam1, v1 = power_iteration(A)
    print(f"\n--- Step 1: Dominant Eigenvalue ---")
    print(f"λ1 = {lam1:.6f}")
    print(f"v1 (normalized) = {np.round(v1, 6)}")

    # Step 2: Deflate
    A2 = hotellings_deflation(A, lam1, v1)
    print(f"\n--- Step 2: Deflated Matrix A2 ---")
    print(np.round(A2, 4))

    # Step 3: Run Power Method on deflated matrix to get λ2
    lam2, v2 = power_iteration(A2)
    print(f"\n--- Step 3: Second Eigenvalue ---")
    print(f"λ2 = {lam2:.6f}")
    print(f"v2 (normalized) = {np.round(v2, 6)}")

    # Step 4: Deflate again for λ3
    A3 = hotellings_deflation(A2, lam2, v2)
    lam3, v3 = power_iteration(A3)
    print(f"\n--- Step 4: Third Eigenvalue ---")
    print(f"λ3 = {lam3:.6f}")

    # NumPy Verification
    evals = np.sort(np.linalg.eigvals(A))[::-1]
    print(f"\n--- NumPy Verification ---")
    print(f"All eigenvalues (sorted): {np.round(evals.real, 6)}")
    print(f"Custom: λ1={lam1:.6f}, λ2={lam2:.6f}, λ3={lam3:.6f}")
