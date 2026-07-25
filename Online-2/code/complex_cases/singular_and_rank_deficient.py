"""
Complex case: singular matrices in LU / inverse power method
=================================================================
Two failure modes worth recognizing on sight:
  1. Naive (no-pivoting) LU on a matrix with a zero LEADING PRINCIPAL MINOR
     -- silently produces inf/nan, no exception.
  2. Inverse power method on a genuinely singular A -- A^-1 doesn't exist,
     LU factorization hits a zero pivot with nothing below it either.
"""
import numpy as np
import warnings


def lu_naive(A, verbose=True):
    A = A.astype(float).copy()
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for k in range(n - 1):
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]     # <-- division by (possibly) zero
            L[i, k] = m
            U[i, k:] -= m * U[k, k:]
    return L, U


def plu_decomposition(A):
    A = A.astype(float).copy()
    n = A.shape[0]
    U = A.copy(); L = np.eye(n); P = np.eye(n)
    fully_singular_columns = []
    for k in range(n - 1):
        p = k
        for i in range(k + 1, n):
            if abs(U[i, k]) > abs(U[p, k]):
                p = i
        if p != k:
            U[[k, p]] = U[[p, k]]; P[[k, p]] = P[[p, k]]
            if k > 0:
                L[[k, p], :k] = L[[p, k], :k]
        if abs(U[k, k]) < 1e-12:
            fully_singular_columns.append(k)
            continue
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            U[i, k:] -= m * U[k, k:]
    return P, L, U, fully_singular_columns


if __name__ == "__main__":
    print("=" * 60, "\n1. Naive LU on a matrix with zero leading minor\n", "=" * 60)
    A1 = np.array([[0, 2, 1], [1, 1, 1], [2, 1, 3]], dtype=float)
    print("A1 =\n", A1, "  (invertible! det =", np.linalg.det(A1), "-- but leading 1x1 minor is 0)")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        L, U = lu_naive(A1)
        for w in caught:
            print(f"  [numpy warning caught] {w.message}")
    print("resulting U (garbage):\n", U)
    print("=> division by exact 0 produces inf, then inf*0 produces nan; no exception raised.")
    print("=> FIX: use PA=LU with partial pivoting instead (see algorithms/lu_decomposition.py).")

    print("\n", "=" * 60, "\n2. PA=LU correctly handles the same matrix\n", "=" * 60)
    P, L, U, singular_cols = plu_decomposition(A1)
    print("P@A1 - L@U (should be ~0):\n", np.round(P @ A1 - L @ U, 10))
    print("fully-singular columns encountered:", singular_cols, "(none -- A1 IS invertible, pivoting fixed it)")

    print("\n", "=" * 60, "\n3. A GENUINELY singular matrix -- pivoting can't save this\n", "=" * 60)
    A2 = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1]], dtype=float)   # row2 = 2*row1
    print("A2 =\n", A2, "  det =", np.linalg.det(A2))
    P, L, U, singular_cols = plu_decomposition(A2)
    print("U =\n", np.round(U, 4))
    print("fully-singular columns encountered:", singular_cols)
    print("=> U has a zero row -- A2 has no inverse, so:")
    print("   - inverse power method cannot run (A^-1 doesn't exist)")
    print("   - Ax=b for this A is either no-solution or infinite-solutions,")
    print("     never unique -- classify via algorithms/gauss_elimination.classify_and_solve")

    print("\nnp.linalg.inv(A2) would raise:", end=" ")
    try:
        np.linalg.inv(A2)
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError: {e}")
