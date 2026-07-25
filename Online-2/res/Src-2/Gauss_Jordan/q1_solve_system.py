import numpy as np

def gauss_jordan(A, b):
    """
    Solves Ax = b using Gauss-Jordan Elimination with Partial Pivoting.
    Drives the coefficient matrix to the Identity matrix (no back substitution needed).
    """
    n = A.shape[0]
    Ab = np.hstack([A, b.reshape(-1, 1)]).astype(float)

    print("\nInitial Augmented Matrix:")
    print(Ab)

    for k in range(n):
        print(f"\n--- Processing Column {k+1} ---")

        # Partial Pivoting
        max_idx = k + np.argmax(np.abs(Ab[k:, k]))
        if max_idx != k:
            print(f"Swapping Row {k+1} and Row {max_idx+1}")
            Ab[[k, max_idx]] = Ab[[max_idx, k]]

        pivot = Ab[k, k]
        if np.abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular. Cannot proceed.")

        # Normalize the pivot row so the diagonal becomes 1
        print(f"Normalizing Row {k+1} by its pivot {pivot:.4f}")
        Ab[k, :] /= pivot

        # Eliminate the variable from EVERY OTHER row (above AND below)
        for i in range(n):
            if i != k:
                factor = Ab[i, k]
                if np.abs(factor) > 1e-15:
                    print(f"  Eliminating Row {i+1} using factor {factor:.4f}")
                Ab[i, :] -= factor * Ab[k, :]

        print("Augmented Matrix:")
        print(np.round(Ab, 4))

    x = Ab[:, -1]

    print("\n--- Result ---")
    print("Final Coefficient Matrix is Identity.")
    print("Solution vector x:", np.round(x, 6))

    return x


def gauss_jordan_inverse(A):
    """
    Computes the inverse of A using Gauss-Jordan Elimination.
    Augments A with the identity matrix and reduces A to I.
    """
    n = A.shape[0]
    AI = np.hstack([A.astype(float), np.eye(n)])

    print("\nInitial Augmented Matrix [A | I]:")
    print(np.round(AI, 4))

    for k in range(n):
        max_idx = k + np.argmax(np.abs(AI[k:, k]))
        if max_idx != k:
            AI[[k, max_idx]] = AI[[max_idx, k]]

        pivot = AI[k, k]
        if np.abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular and cannot be inverted.")

        AI[k, :] /= pivot

        for i in range(n):
            if i != k:
                factor = AI[i, k]
                AI[i, :] -= factor * AI[k, :]

    A_inv = AI[:, n:]
    print("\nReduced [I | A_inv]:")
    print(np.round(AI, 4))
    print("\nInverse A_inv:")
    print(np.round(A_inv, 4))

    return A_inv


if __name__ == "__main__":
    # Part 1: Solve a system
    print("=" * 55)
    print("PART 1: Solve Ax = b using Gauss-Jordan")
    print("=" * 55)
    A = np.array([
        [3, -0.1, -0.2],
        [0.1, 7, -0.3],
        [0.3, -0.2, 10]
    ], dtype=float)
    b = np.array([7.85, -19.3, 71.4], dtype=float)

    print("Matrix A:")
    print(A)
    print("Vector b:", b)
    x = gauss_jordan(A, b)

    np_x = np.linalg.solve(A, b)
    print(f"\nNumPy Verification: {np.round(np_x, 6)}")

    # Part 2: Invert a matrix
    print(f"\n\n{'=' * 55}")
    print("PART 2: Find A_inv using Gauss-Jordan")
    print("=" * 55)
    B = np.array([
        [4, 3, -1],
        [-2, -4, 5],
        [1, 2, 6]
    ], dtype=float)

    print("Matrix B:")
    print(B)
    B_inv = gauss_jordan_inverse(B)

    print("\nVerification (B * B_inv should be I):")
    print(np.round(np.dot(B, B_inv), 4))
