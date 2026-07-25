import numpy as np

def determinant_gauss(A):
    """
    Calculates the determinant of an n×n matrix using Gauss Elimination
    with Partial Pivoting.
    det(A) = (-1)^S * product of diagonal entries of U,
    where S = number of row swaps.
    """
    n = A.shape[0]
    U = A.copy().astype(float)
    num_swaps = 0

    print("\n--- Forward Elimination for Determinant ---")
    for k in range(n - 1):
        max_idx = k + np.argmax(np.abs(U[k:, k]))

        if max_idx != k:
            print(f"Swapping Row {k+1} and Row {max_idx+1}")
            U[[k, max_idx]] = U[[max_idx, k]]
            num_swaps += 1

        if np.abs(U[k, k]) < 1e-12:
            print("Zero pivot encountered. Determinant = 0.")
            return 0.0

        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            U[i, k:] -= factor * U[k, k:]

    print("\nUpper Triangular Matrix (U):")
    print(np.round(U, 4))

    diag_product = np.prod(np.diag(U))
    det_A = ((-1) ** num_swaps) * diag_product

    print(f"\nNumber of row swaps (S): {num_swaps}")
    print(f"Product of U's diagonals: {diag_product:.4f}")
    print(f"(-1)^{num_swaps} × {diag_product:.4f} = {det_A:.4f}")
    print(f"\nDeterminant of A = {det_A:.4f}")

    return det_A


if __name__ == "__main__":
    A = np.array([
        [0, 2, 1],
        [1, -2, -3],
        [-1, 1, 2]
    ], dtype=float)

    print("Matrix A:")
    print(A)

    det_A = determinant_gauss(A)

    print(f"\nNumPy Verification: det(A) = {np.linalg.det(A):.4f}")
