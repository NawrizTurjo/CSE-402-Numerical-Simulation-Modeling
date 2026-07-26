"""
E1 -- Gauss Elimination on a 4x4, with the Determinant as a Byproduct  [~25 min]
================================================================================
Topics: Gauss elimination, partial pivoting, determinant via elimination.
Closest real question: B1 (systems + pivoting). Difficulty: core.

QUESTION
--------
Given
    A = [[ 0,  2,  1, -1],
         [ 3, -1,  2,  4],
         [ 1,  5, -3,  2],
         [ 2,  1,  4, -3]]
    b = [8, 3, 0, 19]

(a) Solve A x = b by Gaussian elimination with partial pivoting, hand-written.
    Print, for every column: the candidate pivots considered, the pivot chosen,
    any row swap, the multiplier used for each row, and the augmented matrix
    after that column is cleared.
(b) Compute det(A) from the SAME elimination -- no second pass -- using
        det(A) = (-1)^(number of row swaps) * product of the diagonal of U.
(c) Verify: report ||Ax - b||_2 (compute it by hand, then cross-check with
    NumPy), compare x against np.linalg.solve and det against np.linalg.det.

WRITTEN QUESTION: A[0][0] is 0 here. Why is that harmless, and what would have
to be true about column 1 for the system to actually be unsolvable?

WHY THIS IS WORTH DRILLING
--------------------------
Bigger than the 3x3s you have practised, so the bookkeeping is where mistakes
appear rather than the algorithm. Also folds in the free determinant -- the
same loop answers two questions, which is exactly how these questions are
usually stacked to fill 30 minutes.
"""
import numpy as np

EPS = 1e-12


def gauss_eliminate_with_determinant(A, b, verbose=True):
    """
    Forward elimination with partial pivoting, tracking swaps for det(A).

    Returns (U_augmented, swap_count).
    """
    n = len(b)
    Aug = [list(A[i]) + [b[i]] for i in range(n)]
    swap_count = 0

    for k in range(n - 1):
        # --- partial pivoting: largest |value| in column k, rows k..n-1 ---
        pivot_row = k
        for i in range(k + 1, n):
            if abs(Aug[i][k]) > abs(Aug[pivot_row][k]):
                pivot_row = i

        if verbose:
            candidates = [round(Aug[i][k], 6) for i in range(k, n)]
            print(f"\n--- column {k + 1} ---")
            print(f"  candidates (rows {k + 1}..{n}): {candidates} "
                  f"-> largest |.| is in row {pivot_row + 1}")

        if pivot_row != k:
            if verbose:
                print(f"  SWAP: R{k + 1} <-> R{pivot_row + 1}   "
                      f"(swap #{swap_count + 1}, flips the sign of det)")
            Aug[k], Aug[pivot_row] = Aug[pivot_row], Aug[k]
            swap_count += 1

        pivot = Aug[k][k]
        if abs(pivot) < EPS:
            if verbose:
                print(f"  pivot is 0 with nothing to swap in -> "
                      "column is dependent, skipping")
            continue
        if verbose:
            print(f"  pivot = {pivot}")

        for i in range(k + 1, n):
            factor = Aug[i][k] / pivot
            if verbose:
                print(f"    R{i + 1} <- R{i + 1} - ({factor:.6f}) * R{k + 1}")
            for j in range(k, n + 1):
                Aug[i][j] -= factor * Aug[k][j]

        if verbose:
            print("  augmented matrix after this column:")
            for row in Aug:
                print("    ", [round(value, 6) for value in row])

    return Aug, swap_count


def back_substitution(Aug):
    """x_i = (c_i - sum_{j>i} U_ij x_j) / U_ii, bottom row upward."""
    n = len(Aug)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = Aug[i][n]
        for j in range(i + 1, n):
            total -= Aug[i][j] * x[j]
        x[i] = total / Aug[i][i]
    return x


def l2_norm(vec):
    """Hand-written ||v||_2."""
    total = 0.0
    for value in vec:
        total += value * value
    return total ** 0.5


def matvec(M, v):
    """Hand-written M @ v."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


if __name__ == "__main__":
    A = [[0, 2, 1, -1],
         [3, -1, 2, 4],
         [1, 5, -3, 2],
         [2, 1, 4, -3]]
    b = [8, 3, 0, 19]
    n = 4

    print("=" * 70)
    print("(a) FORWARD ELIMINATION WITH PARTIAL PIVOTING")
    print("=" * 70)
    print(f"  note A[1][1] = {A[0][0]} -- a zero first pivot, handled by the swap below")
    Aug, swaps = gauss_eliminate_with_determinant(A, b)

    print("\n  back substitution:")
    x = back_substitution(Aug)
    for i in range(n - 1, -1, -1):
        print(f"    x{i + 1} = {x[i]:.10f}")
    print(f"\n  x = {[round(value, 8) for value in x]}")

    print("\n" + "=" * 70)
    print("(b) DETERMINANT FROM THE SAME ELIMINATION")
    print("=" * 70)
    diagonal = [Aug[i][i] for i in range(n)]
    det = (-1.0) ** swaps
    for value in diagonal:
        det *= value
    print(f"  diag(U)   = {[round(value, 6) for value in diagonal]}")
    print(f"  row swaps = {swaps}  ->  sign factor = {(-1) ** swaps}")
    print(f"  det(A) = {(-1) ** swaps} * "
          f"{' * '.join(str(round(value, 6)) for value in diagonal)} = {det:.10f}")
    print("\n  (adding a multiple of one row to another never changes det, so the")
    print("   only correction needed is the sign flip per row swap)")

    print("\n" + "=" * 70)
    print("(c) VERIFICATION")
    print("=" * 70)
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)

    # residual by hand first (it is part of our answer), numpy as cross-check
    Ax = matvec(A, x)
    residual = [Ax[i] - b[i] for i in range(n)]
    manual_res = l2_norm(residual)
    numpy_res = float(np.linalg.norm(A_np @ np.array(x) - b_np))

    print(f"  our x                = {[round(value, 8) for value in x]}")
    print(f"  np.linalg.solve(A,b) = {np.round(np.linalg.solve(A_np, b_np), 8)}")
    print(f"  Ax - b               = {[round(value, 12) for value in residual]}")
    print(f"  manual ||Ax-b||_2 = {manual_res:.3e}   numpy = {numpy_res:.3e}")
    print(f"\n  our det(A)    = {det:.10f}")
    print(f"  np.linalg.det = {np.linalg.det(A_np):.10f}")

    assert np.allclose(x, np.linalg.solve(A_np, b_np), atol=1e-9)
    assert abs(det - np.linalg.det(A_np)) < 1e-8
    assert manual_res < 1e-9
    print("\n  ALL CHECKS PASSED")


# ----------------------------------------------------------------------
# WRITTEN QUESTION -- answer
#
# "A[0][0] is 0. Why is that harmless, and what would have to be true about
#  column 1 for the system to actually be unsolvable?"
#
# A zero in the pivot POSITION is not a property of the system, it is a
# property of the current row ORDER -- and row order carries no mathematical
# information at all, since swapping two equations does not change their
# solution set. Partial pivoting is going to reorder the rows anyway: it looks
# down column 1 for the largest-magnitude entry (3, in row 2, here), swaps it
# up, and proceeds. The zero never gets used as a divisor.
#
# The system is only in trouble if EVERY candidate in that column, at and below
# the current row, is zero -- i.e. the entire column is zero from row k down.
# Then no swap can help, the column contributes no pivot, and some row ends up
# reduced to
#       0*x1 + 0*x2 + 0*x3 + 0*x4 = c
# and the RIGHT-HAND SIDE decides which failure it is:
#   c != 0  ->  a contradiction (0 = 5) -> NO SOLUTION
#   c == 0  ->  a redundant equation (0 = 0), the row was linearly dependent on
#               the others -> INFINITELY MANY SOLUTIONS, with one free variable
#               per missing pivot.
#
# So: a zero pivot is inconclusive on its own; a fully zero ROW after pivoting
# is exhausted is what classifies the system, and that dead row's RHS picks
# between "no solution" and "infinitely many". (This is the same answer as B1's
# written question -- see prev-solutions/gauss_elimination/.)
#
# Practical footnote: pivoting is not only about avoiding division by an exact
# zero. A pivot that is merely SMALL is nearly as bad, because the multipliers
# become huge and amplify round-off -- see practice/a2prep/p2_roundoff_error_
# and_complexity.py, where a 0.003 pivot produces a 67% error in the answer.
# ----------------------------------------------------------------------
