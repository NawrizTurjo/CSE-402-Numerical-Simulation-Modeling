"""
P1a -- Gauss-Jordan RREF & Determinant          [~25 min]  source: res/A2_Prep.md sec. 7
========================================================================================

QUESTION
--------
Given
    0x1 +  x2 + 2x3 =  8
     x1 -  x2 + 3x3 =  8
    2x1 + 4x2 +  x3 = 13
i.e.  A = [[0, 1, 2], [1, -1, 3], [2, 4, 1]],  b = [8, 8, 13]

(a) Reduce [A|b] to REDUCED ROW ECHELON FORM using Gauss-Jordan with partial
    pivoting, by hand. Print per column: candidate pivots, pivot chosen, every
    row swap, and the matrix after the column is cleared BOTH above and below
    the pivot. Read x straight off the final [I|x] -- no back substitution.
(b) Using the SAME elimination, compute det(A) as a byproduct: the product of
    the RAW pivots (before each row is normalized), times (-1)^(row swaps).
(c) Verify against np.linalg.solve(A, b) and np.linalg.det(A).

WRITTEN QUESTION: why must det(A) come from the pivots as they were BEFORE
normalization, rather than from the fully-reduced RREF where every pivot is 1?

WHAT THIS DRILLS
----------------
- Gauss-Jordan differs from Gauss elimination in exactly two places: normalize
  the pivot row, and eliminate ABOVE the pivot as well as below.
- Capturing the determinant mid-sweep, before the information is destroyed.
- Partial pivoting saving a matrix whose very first pivot is 0.
"""
import numpy as np

EPS = 1e-12


def gauss_jordan_rref_with_determinant(A, b, verbose=True):
    """
    Drives [A|b] to [I|x] and collects the determinant on the way.

    Returns (x, det, raw_pivots, swap_count, augmented_matrix).
    """
    n = len(b)
    # [A | b] as plain lists -- one row per equation, RHS tacked on the end
    Aug = [list(A[i]) + [b[i]] for i in range(n)]

    raw_pivots = []
    swap_count = 0

    for col in range(n):
        # --- partial pivoting: largest |value| in this column, at/below row col
        pivot_row = col
        for i in range(col + 1, n):
            if abs(Aug[i][col]) > abs(Aug[pivot_row][col]):
                pivot_row = i

        if verbose:
            candidates = [round(Aug[i][col], 6) for i in range(col, n)]
            print(f"\n--- column {col + 1} ---")
            print(f"  candidate pivots (rows {col + 1}..{n}): {candidates}"
                  f"  -> choose row {pivot_row + 1}")

        if pivot_row != col:
            if verbose:
                print(f"  SWAP: R{col + 1} <-> R{pivot_row + 1}")
            Aug[col], Aug[pivot_row] = Aug[pivot_row], Aug[col]
            swap_count += 1

        # --- RECORD THE RAW PIVOT *NOW*, before normalization destroys it ---
        pivot = Aug[col][col]
        if abs(pivot) < EPS:
            raise ZeroDivisionError(f"column {col} has no usable pivot -- A is singular")
        raw_pivots.append(pivot)
        if verbose:
            print(f"  raw pivot (recorded for det) = {pivot}")

        # --- normalize the pivot row so the pivot becomes exactly 1 ---
        for j in range(n + 1):
            Aug[col][j] = Aug[col][j] / pivot

        # --- eliminate this column from EVERY other row: above AND below.
        #     This is the only structural difference from Gauss elimination.
        for row in range(n):
            if row != col and abs(Aug[row][col]) > EPS:
                factor = Aug[row][col]
                for j in range(n + 1):
                    Aug[row][j] -= factor * Aug[col][j]

        if verbose:
            print("  matrix after clearing this column (above and below):")
            for r in Aug:
                print("    ", [round(value, 6) for value in r])

    # left block is I -> the last column IS the solution
    x = [row[-1] for row in Aug]

    # det(A) = (-1)^swaps * product of RAW pivots
    det = (-1.0) ** swap_count
    for pivot in raw_pivots:
        det *= pivot

    return x, det, raw_pivots, swap_count, Aug


def l2_norm(vec):
    """Hand-written ||v||_2 -- the residual is part of OUR answer, not NumPy's."""
    total = 0.0
    for value in vec:
        total += value * value
    return total ** 0.5


def matvec(M, v):
    """Hand-written M @ v."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


if __name__ == "__main__":
    A = [[0, 1, 2],
         [1, -1, 3],
         [2, 4, 1]]
    b = [8, 8, 13]
    n = 3

    print("=" * 70)
    print("(a) GAUSS-JORDAN -> RREF")
    print("=" * 70)
    x, det, raw_pivots, swaps, Aug = gauss_jordan_rref_with_determinant(A, b)

    print(f"\nleft block is the identity -> solution read directly off [I|x]")
    print(f"  x = {[round(value, 6) for value in x]}   (no back substitution needed)")

    print("\n" + "=" * 70)
    print("(b) DETERMINANT FROM THE RAW PIVOTS")
    print("=" * 70)
    print(f"  raw pivots (before normalization) = {[round(p, 6) for p in raw_pivots]}")
    print(f"  row swaps = {swaps}  ->  sign factor (-1)^{swaps} = {(-1) ** swaps}")
    print(f"  det(A) = {(-1) ** swaps} * "
          f"{' * '.join(str(round(p, 6)) for p in raw_pivots)} = {det}")

    print("\n" + "=" * 70)
    print("(c) VERIFICATION")
    print("=" * 70)
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)

    # residual computed BY HAND first (it is part of our answer), numpy after
    Ax = matvec(A, x)
    residual = [Ax[i] - b[i] for i in range(n)]
    manual_res = l2_norm(residual)
    numpy_res = float(np.linalg.norm(A_np @ np.array(x) - b_np))

    print(f"  our x                = {[round(value, 8) for value in x]}")
    print(f"  np.linalg.solve(A,b) = {np.linalg.solve(A_np, b_np)}")
    print(f"  residual Ax - b      = {[round(value, 12) for value in residual]}")
    print(f"  manual ||Ax-b||_2 = {manual_res:.3e}   numpy = {numpy_res:.3e}")
    print(f"  our det(A)     = {det}")
    print(f"  np.linalg.det  = {np.linalg.det(A_np)}")

    assert np.allclose(x, np.linalg.solve(A_np, b_np), atol=1e-9)
    assert abs(det - np.linalg.det(A_np)) < 1e-8
    assert manual_res < 1e-9
    print("\n  ALL CHECKS PASSED")


# ----------------------------------------------------------------------
# WRITTEN QUESTION -- answer
#
# "Why must the determinant come from the row-echelon pivots as they were
#  BEFORE normalization, rather than being read off the fully-reduced RREF?"
#
# The theorem being used is: the determinant of a triangular (or echelon)
# matrix is the product of its diagonal entries. Two of the three elementary
# row operations behave nicely here:
#
#   - adding a multiple of one row to another   -> det UNCHANGED
#   - swapping two rows                         -> det * (-1)
#   - SCALING a row by a factor p               -> det * p       <-- the problem
#
# Gauss-Jordan's extra step -- dividing each pivot row by its pivot so the
# pivot becomes 1 -- is exactly that third operation. Each normalization
# divides the running determinant by the pivot p. By the time the sweep
# finishes, EVERY pivot has been divided out, so the final RREF has
# prod(diag) = 1 * 1 * 1 = 1 no matter what det(A) actually was. The
# information is not merely hidden, it has been destroyed: the same RREF
# [I|x] comes out of matrices with wildly different determinants.
#
# So the pivot must be captured at the instant it is found, before the
# division -- which is what `raw_pivots.append(pivot)` does above, one line
# before the normalization loop. The determinant is then reassembled as
#
#     det(A) = (-1)^(swaps) * prod(raw pivots)
#
# Plain Gauss elimination does not have this problem (it never normalizes),
# which is why det-via-Gauss can just read prod(diag(U)) at the end.
# ----------------------------------------------------------------------
