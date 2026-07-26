"""
P1b -- Matrix Inverse via Gauss-Jordan AND via LU   [~30 min]  source: res/A2_Prep.md sec. 8
============================================================================================

QUESTION
--------
Using the same matrix as P1a,  A = [[0, 1, 2], [1, -1, 3], [2, 4, 1]]:

(a) Compute A^-1 by Gauss-Jordan elimination on [A | I] (partial pivoting) --
    read the inverse off the right-hand block once the left block reaches I.
(b) Compute A^-1 by LU decomposition: factor A ONCE, then solve A x_j = e_j
    (forward + backward substitution) for each column e_j of the identity, and
    assemble the x_j's as the columns of A^-1.
(c) Verify both agree with each other and with np.linalg.inv(A).

WRITTEN QUESTION: for a large (say 1000x1000) matrix, which method would you
prefer, and why?

WHAT THIS DRILLS
----------------
- The [A|I] -> [I|A^-1] trick: it is literally the same column sweep as
  solving [A|b], with the right-hand block widened from 1 column to n.
- "The identity's columns ARE n right-hand sides" -- the direct generalization
  of C2's reuse-the-factorization idea.
- Getting the PERMUTATION right when combining pivoting with the LU route:
  solve L z = P e_j, never L z = e_j. This is the single most common bug here.
"""
import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------
# (a) INVERSE VIA GAUSS-JORDAN ON [A | I]
# ----------------------------------------------------------------------

def gauss_jordan_inverse(A, verbose=True):
    """
    [A | I]  --Gauss-Jordan-->  [I | A^-1].

    Why it works: the sweep applies elementary row operations E_k ... E_1 with
    (E_k ... E_1) A = I, so E_k ... E_1 IS A^-1. Those same operations applied
    to the identity on the right therefore turn it into A^-1.
    """
    n = len(A)
    # augment: row i gets A's row i, then the i-th row of the identity
    M = [list(A[i]) + [1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]

    for col in range(n):
        # partial pivoting -- searched over the LEFT block only
        pivot_row = col
        for i in range(col + 1, n):
            if abs(M[i][col]) > abs(M[pivot_row][col]):
                pivot_row = i

        if pivot_row != col:
            if verbose:
                print(f"  column {col + 1}: SWAP R{col + 1} <-> R{pivot_row + 1}")
            M[col], M[pivot_row] = M[pivot_row], M[col]

        pivot = M[col][col]
        if abs(pivot) < EPS:
            raise ZeroDivisionError("A is singular -- no inverse exists")
        if verbose:
            print(f"  column {col + 1}: pivot = {pivot:.6g}")

        # normalize the pivot row across ALL 2n columns
        for j in range(2 * n):
            M[col][j] /= pivot

        # clear this column from every other row (above and below)
        for row in range(n):
            if row != col and abs(M[row][col]) > EPS:
                factor = M[row][col]
                for j in range(2 * n):
                    M[row][j] -= factor * M[col][j]

        if verbose:
            print("    [A|I] now:")
            for r in M:
                print("      ", [round(value, 6) for value in r])

    # right-hand block is the inverse
    return [row[n:] for row in M]


# ----------------------------------------------------------------------
# (b) INVERSE VIA LU: FACTOR ONCE, SOLVE n TIMES
# ----------------------------------------------------------------------

def plu_decompose(A, verbose=True):
    """
    Partial-pivoted Doolittle factorization  P A = L U.

    `perm` records the row permutation: perm[i] is which ORIGINAL row now sits
    in position i. That is what lets us permute each e_j correctly later.

    CRITICAL DETAIL: when rows k and p are swapped mid-factorization, the
    multipliers ALREADY STORED in L's earlier columns must be swapped too --
    they belong to the physical rows, which just moved.
    """
    n = len(A)
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [list(row) for row in A]
    perm = list(range(n))

    for k in range(n - 1):
        pivot_row = k
        for i in range(k + 1, n):
            if abs(U[i][k]) > abs(U[pivot_row][k]):
                pivot_row = i

        if pivot_row != k:
            if verbose:
                print(f"  column {k + 1}: SWAP R{k + 1} <-> R{pivot_row + 1}")
            U[k], U[pivot_row] = U[pivot_row], U[k]
            perm[k], perm[pivot_row] = perm[pivot_row], perm[k]
            # swap the multipliers already stored in earlier columns of L
            for j in range(k):
                L[k][j], L[pivot_row][j] = L[pivot_row][j], L[k][j]

        pivot = U[k][k]
        if abs(pivot) < EPS:
            continue

        for i in range(k + 1, n):
            factor = U[i][k] / pivot
            L[i][k] = factor
            for j in range(n):
                U[i][j] -= factor * U[k][j]

        if verbose:
            print(f"  after column {k + 1}:")
            print("    L =", [[round(value, 6) for value in r] for r in L])
            print("    U =", [[round(value, 6) for value in r] for r in U])

    return L, U, perm


def forward_substitution(L, b):
    """Solves L z = b, top row down (L unit lower triangular)."""
    n = len(b)
    z = [0.0] * n
    for i in range(n):
        total = b[i]
        for j in range(i):
            total -= L[i][j] * z[j]
        z[i] = total / L[i][i]
    return z


def backward_substitution(U, z):
    """Solves U x = z, bottom row up."""
    n = len(z)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = z[i]
        for j in range(i + 1, n):
            total -= U[i][j] * x[j]
        x[i] = total / U[i][i]
    return x


def lu_inverse(A, verbose=True):
    """
    A^-1 from ONE factorization: solve A x_j = e_j for each identity column.

    The n columns of the identity are just n right-hand sides -- exactly C2's
    "reuse L, U for multiple b" idea, with b1, b2 replaced by e_1 ... e_n.
    """
    n = len(A)
    L, U, perm = plu_decompose(A, verbose=verbose)   # the ONE O(n^3) step

    columns = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        # P @ e_j : reorder the RHS the same way the rows of A were reordered
        e_permuted = [e[perm[i]] for i in range(n)]
        z = forward_substitution(L, e_permuted)      # cheap O(n^2)
        x = backward_substitution(U, z)              # cheap O(n^2)
        columns.append(x)
        if verbose:
            print(f"  column {j + 1} of A^-1 (solving A x = e{j + 1}) = "
                  f"{[round(value, 6) for value in x]}")

    # columns[j] is column j -> transpose into a row-major matrix
    return [[columns[j][i] for j in range(n)] for i in range(n)], L, U, perm


# ----------------------------------------------------------------------
# hand-written helpers (the error norms are part of OUR answer)
# ----------------------------------------------------------------------

def matmul(X, Y):
    rows, inner, cols = len(X), len(Y), len(Y[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            total = 0.0
            for k in range(inner):
                total += X[i][k] * Y[k][j]
            out[i][j] = total
    return out


def frobenius_norm(M):
    total = 0.0
    for row in M:
        for value in row:
            total += value * value
    return total ** 0.5


def max_abs_diff(X, Y):
    largest = 0.0
    for i in range(len(X)):
        for j in range(len(X[0])):
            if abs(X[i][j] - Y[i][j]) > largest:
                largest = abs(X[i][j] - Y[i][j])
    return largest


if __name__ == "__main__":
    A = [[0, 1, 2],
         [1, -1, 3],
         [2, 4, 1]]
    n = 3

    print("=" * 70)
    print("(a) INVERSE VIA GAUSS-JORDAN ON [A | I]")
    print("=" * 70)
    inv_gj = gauss_jordan_inverse(A)
    print("\n  A^-1 (Gauss-Jordan):")
    for row in inv_gj:
        print("   ", [round(value, 6) for value in row])

    print("\n" + "=" * 70)
    print("(b) INVERSE VIA LU (factor once, solve for each identity column)")
    print("=" * 70)
    inv_lu, L, U, perm = lu_inverse(A)
    print("\n  A^-1 (LU):")
    for row in inv_lu:
        print("   ", [round(value, 6) for value in row])
    print(f"\n  row permutation perm = {perm}  "
          f"(position i now holds original row perm[i])")

    print("\n" + "=" * 70)
    print("(c) VERIFICATION")
    print("=" * 70)
    A_np = np.array(A, dtype=float)
    inv_numpy = np.linalg.inv(A_np)

    # The definitional check needs no library inverse at all: A @ A^-1 == I.
    # Computed BY HAND (matmul + Frobenius norm), then cross-checked in numpy.
    identity_err_gj = frobenius_norm(
        [[matmul(A, inv_gj)[i][j] - (1.0 if i == j else 0.0) for j in range(n)]
         for i in range(n)])
    identity_err_lu = frobenius_norm(
        [[matmul(A, inv_lu)[i][j] - (1.0 if i == j else 0.0) for j in range(n)]
         for i in range(n)])

    print(f"  manual ||A A^-1 - I||_F  (GJ) = {identity_err_gj:.3e}   "
          f"numpy = {np.linalg.norm(A_np @ np.array(inv_gj) - np.eye(n), 'fro'):.3e}")
    print(f"  manual ||A A^-1 - I||_F  (LU) = {identity_err_lu:.3e}   "
          f"numpy = {np.linalg.norm(A_np @ np.array(inv_lu) - np.eye(n), 'fro'):.3e}")

    print(f"\n  numpy A^-1:\n{np.round(inv_numpy, 6)}")
    print(f"\n  max|GJ - LU|    = {max_abs_diff(inv_gj, inv_lu):.3e}")
    print(f"  max|GJ - numpy| = {max_abs_diff(inv_gj, inv_numpy.tolist()):.3e}")
    print(f"  max|LU - numpy| = {max_abs_diff(inv_lu, inv_numpy.tolist()):.3e}")

    assert np.allclose(inv_gj, inv_numpy, atol=1e-10)
    assert np.allclose(inv_lu, inv_numpy, atol=1e-10)
    assert identity_err_gj < 1e-10 and identity_err_lu < 1e-10
    print("\n  ALL CHECKS PASSED")


# ----------------------------------------------------------------------
# WRITTEN QUESTION -- answer
#
# "For a large (say 1000x1000) matrix, which of the two methods would you
#  prefer, and why?"
#
# LU, clearly.
#
# Both methods are O(n^3) overall, so the headline complexity is the same --
# the difference is in the CONSTANT and in what can be reused:
#
#   Gauss-Jordan on [A|I]: every one of the n sweep columns must be applied
#   across all 2n columns of the augmented block, and elimination runs both
#   ABOVE and BELOW each pivot. That works out to roughly 2n^3 flops, and none
#   of the work is separable -- you cannot ask for "just column 7 of A^-1", and
#   you cannot reuse anything if a different right-hand side shows up later.
#
#   LU: one factorization, ~(2/3)n^3 flops, done ONCE. Each column of the
#   inverse is then a pair of triangular solves at ~2n^2 flops, so all n
#   columns add ~2n^3 ... but that part is optional. Everything expensive is
#   in the reusable factorization, so:
#     - need only some columns of A^-1?  pay only for those.
#     - a new right-hand side b arrives? two O(n^2) solves, not a new O(n^3).
#     - need det(A)?  it is already sitting there as (-1)^swaps * prod(diag U).
#
# This is C2's argument again, one level up: the identity's columns are just
# "n right-hand sides", and the point of LU is that right-hand sides are cheap
# once the matrix has been factored.
#
# The practical footnote worth adding: for n = 1000 you should usually not
# form A^-1 AT ALL. If the real goal is to solve A x = b, solving directly via
# LU costs one factorization plus one O(n^2) solve, whereas computing A^-1 and
# then multiplying costs n solves plus a matrix-vector product -- more work,
# and numerically worse (explicit inversion amplifies round-off, and a
# near-singular A gives a wildly inaccurate A^-1 while the direct solve stays
# comparatively well behaved).
# ----------------------------------------------------------------------
