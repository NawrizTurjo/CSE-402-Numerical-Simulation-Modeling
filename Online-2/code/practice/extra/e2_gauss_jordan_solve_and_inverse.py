"""
E2 -- Gauss-Jordan: Solve AND Invert in One Sitting     [~30 min]
=================================================================
Topics: Gauss-Jordan RREF, matrix inverse via [A|I], x = A^-1 b.
Closest real question: none yet (Gauss-Jordan is in the syllabus, untested).
Difficulty: core.

QUESTION
--------
Given
    A = [[ 2, -1,  3],
         [ 4,  2, -1],
         [-2,  3,  5]]
    b = [ 9,  5, 19]

(a) Solve A x = b with Gauss-Jordan elimination (partial pivoting), driving
    [A|b] all the way to [I|x]. Print the matrix after each column is cleared
    above AND below its pivot.
(b) Compute A^-1 by running the same sweep on [A|I].
(c) Recover the solution a second way as x = A^-1 b (hand-written matrix-vector
    product) and confirm it matches (a).
(d) Verify everything against np.linalg.solve and np.linalg.inv, reporting
    ||Ax - b||_2 and ||A A^-1 - I||_F -- both hand-computed first, then
    cross-checked against NumPy.

WRITTEN QUESTION: you now have two routes to x -- Gauss-Jordan on [A|b], and
computing A^-1 then multiplying. Which should you prefer in practice, and why?

WHY THIS IS WORTH DRILLING
--------------------------
Gauss-Jordan is the one syllabus topic with no real past question attached to
it, and the inverse-via-[A|I] variant is the obvious way to ask about it. Both
parts are the same loop, so this is cheap to learn and covers a real gap.
"""
import numpy as np

EPS = 1e-12


def gauss_jordan(A, rhs_block, verbose=True, label="[A|b]"):
    """
    Drives [A | rhs_block] to [I | answer] with partial pivoting.

    `rhs_block` is a list of rows, each holding that row's right-hand entries.
    Pass one column per row to solve a system; pass the identity's rows to get
    the inverse. The loop below does not care which -- that is the whole point.
    """
    n = len(A)
    width = len(rhs_block[0])
    M = [list(A[i]) + list(rhs_block[i]) for i in range(n)]

    for col in range(n):
        # partial pivoting over the LEFT block only
        pivot_row = col
        for i in range(col + 1, n):
            if abs(M[i][col]) > abs(M[pivot_row][col]):
                pivot_row = i

        if verbose:
            print(f"\n--- {label} column {col + 1} ---")
            print(f"  candidates: {[round(M[i][col], 6) for i in range(col, n)]}"
                  f" -> row {pivot_row + 1}")

        if pivot_row != col:
            if verbose:
                print(f"  SWAP R{col + 1} <-> R{pivot_row + 1}")
            M[col], M[pivot_row] = M[pivot_row], M[col]

        pivot = M[col][col]
        if abs(pivot) < EPS:
            raise ZeroDivisionError(f"column {col} has no usable pivot -- A is singular")
        if verbose:
            print(f"  pivot = {pivot}  -> divide R{col + 1} by it so the pivot becomes 1")

        # NORMALIZE across the full width (left block + right block)
        for j in range(n + width):
            M[col][j] /= pivot

        # ELIMINATE above AND below -- this is what makes it Gauss-Jordan
        for row in range(n):
            if row != col and abs(M[row][col]) > EPS:
                factor = M[row][col]
                if verbose:
                    print(f"    R{row + 1} <- R{row + 1} - ({factor:.6f}) * R{col + 1}"
                          f"  {'(ABOVE the pivot)' if row < col else '(below the pivot)'}")
                for j in range(n + width):
                    M[row][j] -= factor * M[col][j]

        if verbose:
            for r in M:
                print("    ", [round(value, 6) for value in r])

    return [row[n:] for row in M]      # the transformed right-hand block


def matvec(M, v):
    """Hand-written M @ v."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


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


def l2_norm(vec):
    total = 0.0
    for value in vec:
        total += value * value
    return total ** 0.5


def frobenius_norm(M):
    total = 0.0
    for row in M:
        for value in row:
            total += value * value
    return total ** 0.5


if __name__ == "__main__":
    A = [[2, -1, 3],
         [4, 2, -1],
         [-2, 3, 5]]
    b = [9, 5, 19]
    n = 3

    print("=" * 70)
    print("(a) GAUSS-JORDAN ON [A | b]  ->  [I | x]")
    print("=" * 70)
    result = gauss_jordan(A, [[b[i]] for i in range(n)], label="[A|b]")
    x = [row[0] for row in result]
    print(f"\n  left block is I, so x is read straight off the right block:")
    print(f"  x = {[round(value, 8) for value in x]}   (no back substitution)")

    print("\n" + "=" * 70)
    print("(b) THE SAME SWEEP ON [A | I]  ->  [I | A^-1]")
    print("=" * 70)
    identity_rows = [[1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]
    A_inv = gauss_jordan(A, identity_rows, verbose=False, label="[A|I]")
    print("  A^-1 =")
    for row in A_inv:
        print("   ", [round(value, 8) for value in row])
    print("\n  (same loop as (a) -- only the right-hand block got wider, from")
    print("   1 column to n. Column j of the answer solves A x = e_j.)")

    print("\n" + "=" * 70)
    print("(c) SOLUTION A SECOND WAY:  x = A^-1 b")
    print("=" * 70)
    x_via_inverse = matvec(A_inv, b)
    print(f"  A^-1 @ b = {[round(value, 8) for value in x_via_inverse]}")
    print(f"  from (a) = {[round(value, 8) for value in x]}")
    print(f"  max difference = "
          f"{max(abs(x[i] - x_via_inverse[i]) for i in range(n)):.3e}")

    print("\n" + "=" * 70)
    print("(d) VERIFICATION")
    print("=" * 70)
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)

    Ax = matvec(A, x)
    residual = [Ax[i] - b[i] for i in range(n)]
    manual_res = l2_norm(residual)
    numpy_res = float(np.linalg.norm(A_np @ np.array(x) - b_np))
    print(f"  manual ||Ax-b||_2 = {manual_res:.3e}   numpy = {numpy_res:.3e}")

    product = matmul(A, A_inv)
    id_diff = [[product[i][j] - (1.0 if i == j else 0.0) for j in range(n)]
               for i in range(n)]
    manual_id = frobenius_norm(id_diff)
    numpy_id = float(np.linalg.norm(A_np @ np.array(A_inv) - np.eye(n), 'fro'))
    print(f"  manual ||A A^-1 - I||_F = {manual_id:.3e}   numpy = {numpy_id:.3e}")

    print(f"\n  our x                = {[round(value, 8) for value in x]}")
    print(f"  np.linalg.solve(A,b) = {np.round(np.linalg.solve(A_np, b_np), 8)}")
    print(f"  our A^-1  =\n{np.round(np.array(A_inv), 6)}")
    print(f"  np.linalg.inv(A) =\n{np.round(np.linalg.inv(A_np), 6)}")

    assert np.allclose(x, np.linalg.solve(A_np, b_np), atol=1e-9)
    assert np.allclose(A_inv, np.linalg.inv(A_np), atol=1e-9)
    assert np.allclose(x, x_via_inverse, atol=1e-9)
    assert manual_res < 1e-9 and manual_id < 1e-9
    print("\n  ALL CHECKS PASSED")


# ----------------------------------------------------------------------
# WRITTEN QUESTION -- answer
#
# "Two routes to x: Gauss-Jordan on [A|b], or compute A^-1 and multiply.
#  Which should you prefer in practice, and why?"
#
# Solve directly. Never form A^-1 just to solve one system.
#
# COST. Gauss-Jordan on [A|b] sweeps a right-hand block one column wide;
# [A|I] sweeps one n columns wide, then you still owe an O(n^2) matrix-vector
# product. Forming the inverse is roughly n times more arithmetic than the
# thing you actually wanted, and every bit of that extra work is thrown away
# after the single multiplication.
#
# ACCURACY -- the more important reason. Every entry of A^-1 is itself the
# result of a full elimination, so it carries its own round-off; multiplying b
# by that noisy matrix then mixes the error from all n columns into every
# component of x. Solving directly keeps b inside the elimination, where the
# error stays bounded by the pivot growth of one sweep. The gap is invisible on
# a well-conditioned 3x3 like this one and severe on an ill-conditioned matrix:
# an explicitly inverted near-singular A can be numerically meaningless while
# the direct solve still returns something usable.
#
# WHEN THE INVERSE *IS* THE RIGHT ANSWER:
#   - the question explicitly asks for A^-1 (as here, and as in A1, where the
#     prompt names np.linalg.inv() as an allowed building block);
#   - you need the entries themselves for a covariance matrix, a sensitivity
#     analysis, or a formula that genuinely contains A^-1;
#   - you have very many right-hand sides AND cannot keep a factorization
#     around -- though even then, keeping L and U is better than keeping A^-1,
#     since it is cheaper to build and more accurate to apply.
#
# Rule of thumb: "x = A^-1 b" is how the maths is WRITTEN; "solve A x = b" is
# how it should be COMPUTED.
# ----------------------------------------------------------------------
