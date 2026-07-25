"""
LU Decomposition (Doolittle) -- naive and with partial pivoting
==================================================================
Covers every LU-flavored question seen in past exams:
  - naive Doolittle LU (no pivoting), printing L/U after each column
  - PA = LU with partial pivoting (hand-written pivot + swap, incl. swapping
    L's already-stored multipliers)
  - forward/backward substitution to solve Ax=b via L,U
  - determinant from U's diagonal + swap parity
  - inverse column-by-column via repeated triangular solves (A^-1 NEVER
    formed explicitly mid-algorithm)

Theory reminders (full version in code/cheatsheet.md):
  - A = LU (no pivoting) exists iff every LEADING PRINCIPAL MINOR is nonzero.
    det(A) != 0 alone is NOT sufficient. Counterexample: [[0,1],[1,0]].
  - PA = LU exists for EVERY invertible matrix (partial pivoting fixes it).
  - Solving with pivoting: solve Lz = P@b, NOT Lz = b.
  - Never form A^-1 just to solve Ax=b -- 2 triangular solves (~2n^2) beats
    forming the inverse first (~2n^3 total, n times more expensive).
"""
import numpy as np


# ------------------------------------------------------------------
# 1. Naive Doolittle LU (no pivoting) -- fails if a pivot is/becomes 0
# ------------------------------------------------------------------
def lu_naive(A, verbose=True):
    """A = L @ U. L unit-lower-triangular, U upper-triangular."""
    A = A.astype(float).copy()
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]        # multiplier -- divides by the pivot
            L[i, k] = m
            U[i, k:] -= m * U[k, k:]
        if verbose:
            print(f"after column {k}:\n  L=\n{np.round(L, 4)}\n  U=\n{np.round(U, 4)}")

    return L, U


# ------------------------------------------------------------------
# 2. PA = LU with partial pivoting (hand-written pivot selection + swap)
# ------------------------------------------------------------------
def plu_decomposition(A, verbose=True):
    """Computes P, L, U, swap_count such that P @ A = L @ U."""
    A = A.astype(float).copy()
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    swap_count = 0

    for k in range(n - 1):
        # ---- hand-written pivot selection ----
        p = k
        for i in range(k + 1, n):
            if abs(U[i, k]) > abs(U[p, k]):
                p = i
        if verbose:
            print(f"column {k}: pivot = {U[p, k]:.6g} (row {p})")

        # ---- hand-written row swap: U, P, AND already-built part of L ----
        if p != k:
            if verbose:
                print(f"  swap: R{k} <-> R{p}")
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            if k > 0:
                L[[k, p], :k] = L[[p, k], :k]   # multipliers move with their row!
            swap_count += 1

        if abs(U[k, k]) < 1e-12:
            continue  # singular column; leave zeros, elimination has nothing to do

        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            U[i, k:] -= m * U[k, k:]

    return P, L, U, swap_count


# ------------------------------------------------------------------
# 3. Triangular solves
# ------------------------------------------------------------------
def forward_substitution(L, b):
    """Solves Lz = b, L unit-lower-triangular (no division needed on diag=1,
    but this also works for a general lower-triangular L)."""
    n = len(b)
    z = np.zeros(n)
    for i in range(n):
        z[i] = (b[i] - L[i, :i] @ z[:i]) / L[i, i]
    return z


def backward_substitution(U, z):
    """Solves Ux = z, U upper-triangular."""
    n = len(z)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def solve_via_lu(A, b, verbose=True):
    """Solves Ax = b using PA = LU. Handles the permutation correctly:
    PAx = Pb -> LUx = Pb -> Lz = Pb, then Ux = z."""
    P, L, U, _ = plu_decomposition(A, verbose=verbose)
    Pb = P @ b
    z = forward_substitution(L, Pb)
    x = backward_substitution(U, z)
    if verbose:
        print("P=\n", P, "\nL=\n", np.round(L, 4), "\nU=\n", np.round(U, 4))
        print("z =", np.round(z, 4), "  x =", np.round(x, 4))
    return x, P, L, U


# ------------------------------------------------------------------
# 4. Determinant from the factorization
# ------------------------------------------------------------------
def lu_determinant(A, verbose=True):
    """det(A) = (-1)^(#row swaps) * product(diagonal of U)."""
    P, L, U, swaps = plu_decomposition(A, verbose=False)
    det = ((-1) ** swaps) * np.prod(np.diag(U))
    if verbose:
        print(f"swaps={swaps}  diag(U)={np.round(np.diag(U), 4)}  det={det:.6g}")
    return det, P, L, U, swaps


# ------------------------------------------------------------------
# 5. Inverse column-by-column (A^-1 never formed mid-algorithm)
# ------------------------------------------------------------------
def lu_inverse(A, verbose=True):
    """A^-1's column j solves A x_j = e_j. Factor ONCE, reuse for every j."""
    n = A.shape[0]
    P, L, U, _ = plu_decomposition(A, verbose=False)

    Ainv = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        Pe = P @ e                      # PAx = Pe -> LUx = Pe
        z = forward_substitution(L, Pe)
        x = backward_substitution(U, z)
        Ainv[:, j] = x                  # x IS column j -- no transpose needed here

    if verbose:
        print("A^-1 =\n", np.round(Ainv, 4))
    return Ainv, P, L, U


if __name__ == "__main__":
    print("=" * 60, "\n1. Naive LU (rocket-fit matrix)\n", "=" * 60)
    A_rocket = np.array([[25, 5, 1], [64, 8, 1], [144, 12, 1]], dtype=float)
    L, U = lu_naive(A_rocket)
    print("||A - L@U||_F =", np.linalg.norm(A_rocket - L @ U, 'fro'))

    print("\n", "=" * 60, "\n2. PA=LU + solve (needs pivoting)\n", "=" * 60)
    A = np.array([[0, 2, 1], [1, 1, 1], [2, 1, 3]], dtype=float)
    b = np.array([5, 3, 6], dtype=float)
    x, P, L, U = solve_via_lu(A, b)
    print("np.linalg.solve check:", np.round(np.linalg.solve(A, b), 4))
    print("||PA - LU||_F =", np.linalg.norm(P @ A - L @ U, 'fro'))

    print("\n", "=" * 60, "\n3. Determinant via LU\n", "=" * 60)
    A_det = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
    det, *_ = lu_determinant(A_det)
    print("np.linalg.det check:", np.linalg.det(A_det))

    print("\n", "=" * 60, "\n4. Inverse via LU\n", "=" * 60)
    Ainv, *_ = lu_inverse(A_det)
    print("np.linalg.inv check:\n", np.round(np.linalg.inv(A_det), 4))
    print("||A@Ainv - I||_F =", np.linalg.norm(A_det @ Ainv - np.eye(3), 'fro'))


# ----------------------------------------------------------------------
# Written-question quick answers (full derivations in code/cheatsheet.md)
# ----------------------------------------------------------------------
# Q: Why does storing m = U[i,k]/U[k,k] directly into L[i,k] give a valid A=LU?
# A: Elimination = left-multiplying by elementary matrices E_n...E_1 A = U.
#    A = (E_1^-1...E_n^-1) U, and that product collapses into a single unit
#    lower-triangular matrix with the multipliers sitting exactly in their
#    (i,k) slots -- no matrix inversion/multiplication needed at runtime.
#
# Q: Cost: LU factor vs one solve vs Gauss elimination repeated for k RHS?
# A: Factor once: O(n^3) (~2n^3/3 flops). Each solve (2 triangular passes):
#    O(n^2). k different b's: O(n^3 + k*n^2) via LU vs O(k*n^3) via repeated
#    Gauss elimination -- LU wins whenever k > ~1.
#
# Q: When would you use Cholesky (A = L L^T) instead of general LU?
# A: A symmetric positive-definite -> ~n^3/3 flops (half the cost), no
#    pivoting needed (SPD guarantees positive pivots throughout).
#
# Q: Small ||Ax-b|| but x looks "wrong" compared to a friend's answer?
# A: Ill-conditioning. kappa(A) = ||A||*||A^-1||; relative error in x can be
#    ~ kappa(A) * relative residual, even when the residual itself is tiny.
