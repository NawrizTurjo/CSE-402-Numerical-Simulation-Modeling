import numpy as np

def gauss_elimination_partial_pivoting(A, b):
    """
    Solves Ax = b using Gauss Elimination with Partial Pivoting.
    Prints all row swaps, chosen pivots, and the augmented matrix after forward elimination.
    """
    n = A.shape[0]
    Ab = np.hstack([A, b.reshape(-1, 1)]).astype(float)

    print("\nInitial Augmented Matrix:")
    print(Ab)
    print("\n--- Forward Elimination with Partial Pivoting ---")

    for k in range(n - 1):
        # Search column k (from row k to n) for the absolute largest value
        max_idx = k + np.argmax(np.abs(Ab[k:, k]))
        pivot_val = Ab[max_idx, k]

        print(f"\nStep {k+1}: Column {k+1}")
        print(f"  Chosen Pivot = {pivot_val:.6f} (at Row {max_idx+1})")

        # Swap rows if necessary
        if max_idx != k:
            print(f"  -> Swapping Row {k+1} and Row {max_idx+1}")
            Ab[[k, max_idx]] = Ab[[max_idx, k]]

        # Check for zero pivot
        if np.abs(Ab[k, k]) < 1e-12:
            print("  -> Warning: Pivot is effectively zero. Matrix may be singular.")
            continue

        # Eliminate entries below the pivot
        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]

    print("\nAugmented Matrix after Forward Elimination:")
    print(np.round(Ab, 6))

    # Back Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if np.abs(Ab[i, i]) < 1e-12:
            raise ValueError("Matrix is singular. Cannot perform back substitution.")
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]

    return x


if __name__ == "__main__":
    A = np.array([
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ], dtype=float)
    b = np.array([8, -11, -3], dtype=float)

    print("Matrix A:")
    print(A)
    print("Vector b:", b)

    x = gauss_elimination_partial_pivoting(A, b)

    print("\n--- Result ---")
    print(f"Solution vector x: {np.round(x, 6)}")

    # NumPy Verification
    np_x = np.linalg.solve(A, b)
    print(f"NumPy Verification: {np.round(np_x, 6)}")

    # L2 Norm
    residual = np.dot(A, x) - b
    print(f"||Ax - b||_2 = {np.linalg.norm(residual):.6e}")
