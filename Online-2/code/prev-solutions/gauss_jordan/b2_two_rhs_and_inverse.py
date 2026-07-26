"""
B2's online question -- augmented [A | b1 | b2], solved by Gauss-Jordan, then
the SAME sweep repurposed on [A | I] to get A^-1
====================================================================================

EXACT DATA, as given in res/Online Questions.txt section 5:

    A  = [[1, 1, 1],       b1 = [1, 2, 1]      b2 = [2, 1, -1]
          [1, 2, 3],
          [1, 3, 6]]

The question, verbatim in structure:
  1. Construct the augmented matrix [A | b1 | b2].
  2. Run Gauss-Jordan on the augmented matrix. No row swapping is necessary.
  3. Find the solution vectors x1 and x2.
  4. Replace b1|b2 with I so the right side becomes A^-1. Use the SAME
     Gauss-Jordan function.
  5. Verify the solutions against np.linalg.solve and the inverse against
     np.linalg.inv.

This is C2's "reuse across right-hand sides" idea (algorithms/lu_decomposition.py
`lu_solve_multiple_rhs`) applied to Gauss-Jordan instead of LU, and it is
literally the same trick that produces the matrix inverse: augmenting with I
is just augmenting with n right-hand sides, e_1 ... e_n, instead of 2. One
function -- `gauss_jordan_multi` below -- handles both b1|b2 (2 columns) and
I (3 columns) without any change; only the width of the right-hand block
differs.

Every norm and residual below is HAND-COMPUTED first and cross-checked against
NumPy after -- they are part of the answer, not verification (res/A2_Prep.md
section 1). Both numbers are printed and must agree.
"""
import numpy as np

EPS = 1e-9


# ----------------------------------------------------------------------
# THE ONE FUNCTION -- used for both [A|b1|b2] and [A|I]
# ----------------------------------------------------------------------

def gauss_jordan_multi(A, B, partial_pivot=False, verbose=True):
    """
    Drives [A | B] to [I | X] by Gauss-Jordan, where B (and therefore X) can
    have ANY number of columns -- 1 (a single b), 2 (b1 and b2 side by side),
    or n (the identity, giving A^-1). The column sweep never looks at how wide
    the right-hand block is; it just sweeps every column of the augmented row.

    `partial_pivot=False` by default because the question states no row
    swapping is necessary for THIS A: every diagonal entry stays nonzero
    through the whole sweep, so picking the largest |entry| in the column
    (the general, numerically-safer default used in algorithms/gauss_jordan.py)
    is not required for correctness here -- set partial_pivot=True to get that
    stability behaviour back for a matrix that doesn't have this guarantee.

    Returns:
        X (list[list]): solution block, same shape as B.
        M (list[list]): final augmented matrix [I | X].
    """
    n = len(A)
    m = len(B[0])
    # augment: row i gets A's row i, then B's row i
    M = [list(A[i]) + list(B[i]) for i in range(n)]

    for col in range(n):
        pivot_row = col
        if partial_pivot:
            # search over the LEFT (A) block only
            for i in range(col + 1, n):
                if abs(M[i][col]) > abs(M[pivot_row][col]):
                    pivot_row = i

        if pivot_row != col:
            if verbose:
                print(f"  column {col + 1}: SWAP R{col + 1} <-> R{pivot_row + 1}")
            M[col], M[pivot_row] = M[pivot_row], M[col]

        pivot = M[col][col]
        if abs(pivot) < EPS:
            raise ZeroDivisionError(f"pivot at column {col + 1} is ~0 -- A is singular")

        # normalize the pivot row across ALL n+m columns
        for j in range(n + m):
            M[col][j] /= pivot

        # clear this column from every other row (above AND below -- Gauss-Jordan)
        for row in range(n):
            if row != col and abs(M[row][col]) > EPS:
                factor = M[row][col]
                for j in range(n + m):
                    M[row][j] -= factor * M[col][j]

        if verbose:
            print(f"  after clearing column {col + 1}:")
            for r in M:
                print("    ", [round(value, 6) for value in r])

    X = [row[n:] for row in M]
    return X, M


# ----------------------------------------------------------------------
# hand-written helpers (residuals/products used in verification are OUR
# answer, not something to hand to np.linalg -- see res/A2_Prep.md section 1)
# ----------------------------------------------------------------------

def matvec(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


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


def l2_norm(v):
    total = 0.0
    for value in v:
        total += value * value
    return total ** 0.5


def frobenius_norm(M):
    total = 0.0
    for row in M:
        for value in row:
            total += value * value
    return total ** 0.5


if __name__ == "__main__":
    A = [[1, 1, 1],
         [1, 2, 3],
         [1, 3, 6]]
    b1 = [1, 2, 1]
    b2 = [2, 1, -1]
    n = 3

    # ---- 1 & 2. Augment [A | b1 | b2], run Gauss-Jordan ----
    print("=" * 70)
    print("1 & 2. GAUSS-JORDAN ON [A | b1 | b2]")
    print("=" * 70)
    X, M = gauss_jordan_multi(A, [[b1[i], b2[i]] for i in range(n)])

    # ---- 3. Solution vectors ----
    print("\n" + "=" * 70)
    print("3. SOLUTION VECTORS")
    print("=" * 70)
    x1 = [X[i][0] for i in range(n)]
    x2 = [X[i][1] for i in range(n)]
    print(f"  x1 (solves A x = b1) = {[round(v, 6) for v in x1]}")
    print(f"  x2 (solves A x = b2) = {[round(v, 6) for v in x2]}")

    # ---- 4. Replace b1|b2 with I, run the SAME function ----
    print("\n" + "=" * 70)
    print("4. SAME FUNCTION, RIGHT BLOCK REPLACED BY I  ->  A^-1")
    print("=" * 70)
    I3 = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    A_inv, _ = gauss_jordan_multi(A, I3, verbose=False)
    print("\n  A^-1 =")
    for row in A_inv:
        print("   ", [round(v, 6) for v in row])

    # ---- 5. Verify everything against NumPy ----
    print("\n" + "=" * 70)
    print("5. VERIFICATION")
    print("=" * 70)
    A_np = np.array(A, dtype=float)

    for name, b, x in [("x1", b1, x1), ("x2", b2, x2)]:
        Ax = matvec(A, x)                                    # manual A @ x
        residual = [Ax[i] - b[i] for i in range(n)]          # manual residual
        manual_res = l2_norm(residual)                       # manual ||.||_2
        x_np = np.linalg.solve(A_np, np.array(b, dtype=float))
        numpy_res = float(np.linalg.norm(A_np @ np.array(x) - np.array(b, dtype=float)))
        print(f"  {name}: manual ||Ax-b|| = {manual_res:.3e}   numpy = {numpy_res:.3e}"
              f"   gap vs np.linalg.solve = {np.linalg.norm(np.array(x) - x_np):.3e}")
        assert manual_res < 1e-8
        assert np.allclose(x, x_np, atol=1e-8)

    identity_err = frobenius_norm(
        [[matmul(A, A_inv)[i][j] - (1.0 if i == j else 0.0) for j in range(n)]
         for i in range(n)])                                 # manual ||A A^-1 - I||_F
    numpy_identity_err = float(np.linalg.norm(A_np @ np.array(A_inv) - np.eye(n), 'fro'))
    inv_np = np.linalg.inv(A_np)
    print(f"\n  A^-1: manual ||A A^-1 - I||_F = {identity_err:.3e}"
          f"   numpy = {numpy_identity_err:.3e}"
          f"   gap vs np.linalg.inv = {np.linalg.norm(np.array(A_inv) - inv_np):.3e}")
    assert identity_err < 1e-8
    assert np.allclose(A_inv, inv_np, atol=1e-8)

    print("\n  ALL CHECKS PASSED")


# ----------------------------------------------------------------------
# WRITTEN QUESTIONS -- answers
#
# "Why does using A|b1|b2 work with Gauss-Jordan to solve both systems of
#  equations?"
#
# The row operations Gauss-Jordan performs (swap, scale-by-pivot, eliminate)
# are determined ENTIRELY by the LEFT block (A) -- which row to pivot on and
# what multiple of the pivot row to subtract never look at the right-hand
# columns at all. So driving [A | b1 | b2] to [I | x1 | x2] applies the exact
# same sequence of elementary row operations E_k...E_1 to b1's column and to
# b2's column independently and simultaneously: E_k...E_1 A = I means
# E_k...E_1 b1 = x1 and, in the same pass, E_k...E_1 b2 = x2. Nothing about b1
# ever touches b2's column or vice versa -- they just ride along in separate
# columns through the identical sweep.
#
# "What if both systems of equations have different coefficient matrices, can
#  that be solved in this way?"
#
# No. The entire trick depends on ONE set of row operations solving both
# systems, and that set is fixed by A. If b1 belongs to A1 x = b1 and b2
# belongs to a DIFFERENT A2 x = b2, there is no single sequence of row
# operations that reduces both A1 and A2 to I at once (in general they need
# different pivots, different multipliers, even different numbers of swaps).
# Each system would need to be eliminated separately -- you are back to two
# independent O(n^3) Gauss-Jordan runs.
#
# "Is it necessarily twice as expensive than solving one system of equations
#  if Gauss-Jordan is used?"
#
# No -- it is barely more expensive at all, and the extra cost is NOT another
# full elimination. The pivot search, the swaps, and the elimination
# multipliers are computed once, from the A block only, and reused for every
# extra column. Only the O(n) "apply this multiplier to my column too" step
# repeats per right-hand side, i.e. each extra RHS costs an extra O(n^2), not
# an extra O(n^3). For n=3 with 2 RHS this is roughly (n^2)(n+2) = 45 vs.
# (n^2)(n+1) = 36 multiply-adds -- about 25% more, not 100% more. This is
# exactly C2's LU-reuse argument (prev-solutions/lu_decomposition/
# c2_lu_two_rhs_reuse_factorization.py), just for Gauss-Jordan instead of LU:
# the expensive part (finding the row operations) depends only on A and is
# paid once; each right-hand side only pays for being carried through.
#
# "What matrix is replaced by b1|b2 to get A^-1? Why does it work?"
#
# b1|b2 is replaced by the n x n IDENTITY matrix I. It works because
# augmenting with a wider right-hand block changes nothing about the argument
# above -- the identity's n columns e_1, ..., e_n are just n more right-hand
# sides. Column j of the answer solves A x_j = e_j, and by definition A^-1 is
# exactly the matrix whose j-th column solves that equation for every j
# (A A^-1 = I, read one column at a time). Equivalently: the row operations
# E_k...E_1 satisfy E_k...E_1 A = I, i.e. E_k...E_1 IS A^-1 by definition --
# applying those same operations to the identity on the right-hand side is
# precisely how the sweep computes A^-1 as a byproduct.
# ----------------------------------------------------------------------
