"""
P4 -- Full Eigendecomposition & Matrix Powers    [~30 min]  source: res/A2_Prep.md sec. 11
==========================================================================================

QUESTION
--------
Using A = [[6, 2], [2, 3]] and the eigenpairs found in P3
(lambda1 = 7, v1 = [2/sqrt5, 1/sqrt5];  lambda2 = 2, v2 = [1/sqrt5, -2/sqrt5]):

(a) Build V = [v1 | v2] (eigenvectors as COLUMNS) and Lambda = diag(l1, l2).
(b) Verify A = V Lambda V^-1.
(c) Take x = [3, 1]. Convert to eigen-coordinates via V^-1 x, then back via
    V(...), confirming you recover the original x.
(d) Compute A^5 two ways -- via V Lambda^5 V^-1 (raising the DIAGONAL to a
    power, trivial) and via np.linalg.matrix_power(A, 5). Confirm they agree.

WHAT THIS DRILLS
----------------
- The column convention. V's columns are eigenvectors, and column j must line
  up with Lambda[j][j]. Getting this backwards is the classic silent failure:
  the code runs, the shapes are right, the reconstruction is garbage.
- WHY A^k is cheap: every interior V^-1 V collapses to I, so
      A^k = (V Lam V^-1)^k = V Lam^k V^-1
  and Lam^k is n scalar exponentiations, not a single matrix multiply.
- V^-1 and V as a pair of coordinate translators: V^-1 takes a vector INTO the
  eigenbasis (where A is just "scale each axis"), V brings it back out.
"""
import math
import numpy as np


def inverse_2x2(M):
    """
    Closed-form 2x2 inverse -- hand-computed, no library call needed:
        [[a, b], [c, d]]^-1 = (1/(ad - bc)) [[d, -b], [-c, a]]
    """
    a, b = M[0][0], M[0][1]
    c, d = M[1][0], M[1][1]
    det = a * d - b * c
    if abs(det) < 1e-12:
        raise ValueError("V is singular -- A is not diagonalizable")
    return [[d / det, -b / det],
            [-c / det, a / det]]


def matmul(X, Y):
    """Hand-written X @ Y."""
    rows, inner, cols = len(X), len(Y), len(Y[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            total = 0.0
            for k in range(inner):
                total += X[i][k] * Y[k][j]
            out[i][j] = total
    return out


def matvec(M, v):
    """Hand-written M @ v."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def frobenius_norm(M):
    """Hand-written ||M||_F."""
    total = 0.0
    for row in M:
        for value in row:
            total += value * value
    return total ** 0.5


if __name__ == "__main__":
    A = [[6, 2],
         [2, 3]]
    lam1, lam2 = 7.0, 2.0
    v1 = [2 / math.sqrt(5), 1 / math.sqrt(5)]
    v2 = [1 / math.sqrt(5), -2 / math.sqrt(5)]

    print("=" * 70)
    print("(a) BUILD V (EIGENVECTORS AS COLUMNS) AND Lambda")
    print("=" * 70)
    # v1 goes in COLUMN 0, v2 in COLUMN 1 -- so row i takes the i-th entry of each
    V = [[v1[0], v2[0]],
         [v1[1], v2[1]]]
    Lam = [[lam1, 0.0],
           [0.0, lam2]]

    print(f"  v1 = {[round(value, 6) for value in v1]}  (lambda1 = {lam1})")
    print(f"  v2 = {[round(value, 6) for value in v2]}  (lambda2 = {lam2})")
    print(f"\n  V = {[[round(value, 6) for value in row] for row in V]}   "
          "<- columns are the eigenvectors")
    print(f"  Lambda = {Lam}   <- Lambda[j][j] pairs with V's COLUMN j")

    V_inv = inverse_2x2(V)
    print(f"  V^-1 (closed form) = {[[round(value, 6) for value in row] for row in V_inv]}")
    print(f"  numpy V^-1         = {np.round(np.linalg.inv(np.array(V)), 6).tolist()}")

    print("\n" + "=" * 70)
    print("(b) VERIFY A = V Lambda V^-1")
    print("=" * 70)
    reconstructed = matmul(matmul(V, Lam), V_inv)
    print(f"  V Lambda V^-1 = {[[round(value, 10) for value in row] for row in reconstructed]}")
    print(f"  A             = {A}")

    diff = [[A[i][j] - reconstructed[i][j] for j in range(2)] for i in range(2)]
    manual_err = frobenius_norm(diff)
    numpy_err = float(np.linalg.norm(
        np.array(A, dtype=float) - np.array(V) @ np.array(Lam) @ np.array(V_inv), 'fro'))
    print(f"  manual ||A - V Lam V^-1||_F = {manual_err:.3e}   numpy = {numpy_err:.3e}")

    print("\n" + "=" * 70)
    print("(c) EIGEN-COORDINATE ROUND TRIP")
    print("=" * 70)
    x = [3.0, 1.0]
    # V^-1 x : express x as a combination of the eigenvectors
    c = matvec(V_inv, x)
    # V c : translate that combination back to standard coordinates
    x_back = matvec(V, c)

    print(f"  x (standard coords)      = {x}")
    print(f"  c = V^-1 x (eigen coords) = {[round(value, 6) for value in c]}")
    print(f"      meaning  x = {c[0]:.6f} * v1  +  {c[1]:.6f} * v2")
    print(f"  V c (back to standard)   = {[round(value, 10) for value in x_back]}")
    print(f"  round-trip error         = "
          f"{max(abs(x_back[i] - x[i]) for i in range(2)):.3e}")

    # In eigen-coordinates, applying A is just scaling each axis by its own
    # eigenvalue -- no matrix multiplication involved at all.
    Ax_direct = matvec(A, x)
    Ax_via_eigen = matvec(V, [lam1 * c[0], lam2 * c[1]])
    print(f"\n  A @ x directly              = {[round(value, 8) for value in Ax_direct]}")
    print(f"  V @ (Lambda * eigen coords) = {[round(value, 8) for value in Ax_via_eigen]}")
    print("    -> in the eigenbasis, 'apply A' collapses to 'scale coordinate j "
          "by lambda_j'")

    print("\n" + "=" * 70)
    print("(d) A^5 VIA V Lambda^5 V^-1")
    print("=" * 70)
    k = 5
    # Raising a DIAGONAL matrix to a power = raising each diagonal entry.
    # Two scalar exponentiations; no matrix multiplication for any k.
    Lam_k = [[lam1 ** k, 0.0],
             [0.0, lam2 ** k]]
    print(f"  Lambda^{k} = {[[round(value, 4) for value in row] for row in Lam_k]}   "
          f"(just {lam1}^{k} and {lam2}^{k})")

    A_k = matmul(matmul(V, Lam_k), V_inv)
    A_k_numpy = np.linalg.matrix_power(np.array(A, dtype=float), k)

    print(f"\n  A^{k} via V Lambda^{k} V^-1     = "
          f"{[[round(value, 6) for value in row] for row in A_k]}")
    print(f"  A^{k} via np.linalg.matrix_power = {A_k_numpy.tolist()}")

    err_k = frobenius_norm([[A_k[i][j] - A_k_numpy[i][j] for j in range(2)]
                            for i in range(2)])
    print(f"  manual ||difference||_F = {err_k:.3e}   "
          f"relative = {err_k / np.linalg.norm(A_k_numpy, 'fro'):.3e}")

    assert manual_err < 1e-10
    assert max(abs(x_back[i] - x[i]) for i in range(2)) < 1e-10
    assert np.allclose(A_k, A_k_numpy, rtol=1e-10)
    print("\n  ALL CHECKS PASSED")


# ----------------------------------------------------------------------
# NOTES WORTH REMEMBERING
#
# WHY A^k = V Lambda^k V^-1:
#     A^k = (V Lam V^-1)(V Lam V^-1) ... (V Lam V^-1)          [k copies]
#         = V Lam (V^-1 V) Lam (V^-1 V) ... Lam V^-1
#         = V Lam^k V^-1
# Every interior V^-1 V is the identity and vanishes. Only two matrix
# multiplications remain, no matter how large k is -- plus n scalar powers for
# Lambda^k. Repeated squaring of A would cost O(n^3 log k); this is O(n^3)
# once, then O(n) per additional power.
#
# It also works for k that repeated multiplication cannot express at all:
#   k = -1   -> A^-1 (reciprocate the eigenvalues)
#   k = 0.5  -> a matrix square root (square-root the eigenvalues)
#   k = 100  -> same cost as k = 5
#
# THE COLUMN CONVENTION (the bug to avoid):
#   V's COLUMNS are eigenvectors -- V[:, j] pairs with Lambda[j][j].
#   np.linalg.eig follows the same convention: eigvecs[:, j] goes with
#   eigvals[j]. Building V from eigenvectors as ROWS gives a V that is V^T,
#   and the reconstruction silently produces the wrong matrix.
#   Use np.column_stack((v1, v2)) rather than np.array([v1, v2]).
#
# WHEN THIS BREAKS:
#   A must be DIAGONALIZABLE -- it needs n linearly independent eigenvectors,
#   or V is singular and V^-1 does not exist. A symmetric matrix always
#   qualifies (spectral theorem), which is why every eigen question in this
#   course uses one. A "defective" matrix like [[2,1],[0,2]] has the repeated
#   eigenvalue 2 with only ONE independent eigenvector, so V is singular and
#   there is no eigendecomposition -- inverse_2x2 above raises on exactly that.
# ----------------------------------------------------------------------
