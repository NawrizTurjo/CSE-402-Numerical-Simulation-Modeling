"""
Gauss-Jordan Elimination
=========================
Drives the augmented matrix all the way to [I | x] -- no back substitution.
Two upgrades over plain Gauss elimination:
  1. Normalize each row by its own pivot (pivot becomes 1).
  2. Eliminate the pivot variable from EVERY other row (above AND below).

Includes partial pivoting (recommended -- same stability reasons as Gauss
elimination) and prints the augmented matrix after each column is cleared.
"""
import numpy as np

EPS = 1e-9


def gauss_jordan(A, b, verbose=True, partial_pivot=True):
    n = len(b)
    Aug = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for k in range(n):
        if partial_pivot:
            p = k + int(np.argmax(np.abs(Aug[k:, k])))
            if p != k:
                if verbose:
                    print(f"  row swap: R{k} <-> R{p}")
                Aug[[k, p]] = Aug[[p, k]]

        if abs(Aug[k, k]) < EPS:
            raise ZeroDivisionError(f"pivot at column {k} is ~0 -- matrix is singular")

        Aug[k] = Aug[k] / Aug[k, k]              # normalize pivot row to 1
        for i in range(n):
            if i != k:
                Aug[i] -= Aug[i, k] * Aug[k]      # eliminate from every other row

        if verbose:
            print(f"\nAfter clearing column {k}:\n{np.round(Aug, 4)}")

    return Aug[:, -1], Aug


if __name__ == "__main__":
    A = np.array([[3, -0.1, -0.2],
                  [0.1, 7, -0.3],
                  [0.3, -0.2, 10]], dtype=float)
    b = np.array([7.85, -19.3, 71.4], dtype=float)

    x, Aug_final = gauss_jordan(A, b)

    print("\n>>> Solution x =", np.round(x, 4))         # expected [3, -2.5, 7]
    print("np.linalg.solve  =", np.round(np.linalg.solve(A, b), 4))
    print("||Ax - b||_2     =", np.linalg.norm(A @ x - b))


# ----------------------------------------------------------------------
# Written-question quick answers
# ----------------------------------------------------------------------
# Q: Cost of Gauss-Jordan vs. Gauss elimination?
# A: Gauss-Jordan does strictly MORE arithmetic (eliminates above the pivot
#    too, and normalizes every row), but needs no separate back-substitution
#    pass. Same pitfalls (zero pivot, round-off) apply -- partial pivoting
#    fixes both, exactly as in plain Gauss elimination.
