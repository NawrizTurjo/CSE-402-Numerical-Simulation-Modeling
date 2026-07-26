"""
E3 -- PA = LU: When Naive Doolittle Breaks, and Three RHS to Reuse It On  [~35 min]
===================================================================================
Topics: LU with partial pivoting, permutation handling, multi-RHS reuse, det via LU.
Closest real question: C2 (LU + two right-hand sides). Difficulty: core, hardest variant.

QUESTION
--------
Given
    A  = [[ 1,  2,  3],
          [ 2,  4,  7],
          [ 3,  5,  3]]
    b1 = [ 6, 13, 11],   b2 = [1, 0, 0],   b3 = [2, -1, 4]

(a) Attempt naive Doolittle LU (no pivoting) and show precisely where and why
    it fails on this matrix.
(b) Redo it with partial pivoting to get P A = L U. Print P, L, U and verify
    ||P A - L U||_F -- note it is PA, not A, that equals LU.
(c) Solve A x = b for ALL THREE right-hand sides, reusing the single L, U.
    Remember: solve L z = P b, never L z = b.
(d) Compute det(A) from the factorization as (-1)^swaps * prod(diag(U)).
(e) Verify all three solutions against np.linalg.solve and report each
    residual ||Ax - b||_2, hand-computed and then cross-checked.

WRITTEN QUESTION: naive Doolittle failed here even though A is perfectly
non-singular (det != 0). Explain the difference between "the matrix has no LU
factorization" and "the matrix has no solution", and state what P is doing.

WHY THIS IS WORTH DRILLING
--------------------------
C2's real matrix happened not to need pivoting, so a no-pivot Doolittle passed.
This one has a2 = 2*a1 in the first two rows of column 2, which zeroes the
second pivot and makes naive LU divide by zero. If the exam matrix does that,
the fix is PA = LU plus the P @ b step -- and forgetting P @ b is the single
most common way to get a confident, wrong answer.
"""
import numpy as np

EPS = 1e-12


def naive_doolittle(A, verbose=True):
    """
    Doolittle LU with NO pivoting. Raises when a zero pivot appears.

    This is the version that matches the theory-class derivation, and the one
    that breaks on any matrix whose leading principal minors are not all
    nonzero -- which has nothing to do with the matrix being singular.
    """
    n = len(A)
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [list(row) for row in A]

    for k in range(n - 1):
        pivot = U[k][k]
        if verbose:
            print(f"  column {k + 1}: pivot U[{k + 1}][{k + 1}] = {pivot}")
        if abs(pivot) < EPS:
            raise ZeroDivisionError(
                f"naive Doolittle fails: pivot U[{k + 1}][{k + 1}] = {pivot}. "
                "No pivoting means there is no way to bring a nonzero entry up.")
        for i in range(k + 1, n):
            factor = U[i][k] / pivot
            L[i][k] = factor
            for j in range(n):
                U[i][j] -= factor * U[k][j]
        if verbose:
            print(f"    U now = {[[round(value, 6) for value in r] for r in U]}")
    return L, U


def plu_decompose(A, verbose=True):
    """
    Partial-pivoted Doolittle:  P A = L U.

    THREE THINGS MOVE ON A SWAP, and forgetting the third is the classic bug:
      1. the rows of U,
      2. the rows of P,
      3. the multipliers ALREADY STORED in L's earlier columns -- they belong
         to physical rows, and those rows just moved.
    """
    n = len(A)
    U = [list(row) for row in A]
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    P = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    swaps = 0

    for k in range(n - 1):
        pivot_row = k
        for i in range(k + 1, n):
            if abs(U[i][k]) > abs(U[pivot_row][k]):
                pivot_row = i

        if verbose:
            print(f"\n  column {k + 1}: candidates "
                  f"{[round(U[i][k], 6) for i in range(k, n)]} -> row {pivot_row + 1}")

        if pivot_row != k:
            if verbose:
                print(f"    SWAP R{k + 1} <-> R{pivot_row + 1}  "
                      "(in U, in P, AND in L's stored multipliers)")
            U[k], U[pivot_row] = U[pivot_row], U[k]
            P[k], P[pivot_row] = P[pivot_row], P[k]
            for j in range(k):                       # <-- the easily-forgotten one
                L[k][j], L[pivot_row][j] = L[pivot_row][j], L[k][j]
            swaps += 1

        pivot = U[k][k]
        if abs(pivot) < EPS:
            if verbose:
                print("    column is entirely zero below here -- A is singular")
            continue

        for i in range(k + 1, n):
            factor = U[i][k] / pivot
            L[i][k] = factor
            for j in range(n):
                U[i][j] -= factor * U[k][j]

        if verbose:
            print(f"    L = {[[round(value, 6) for value in r] for r in L]}")
            print(f"    U = {[[round(value, 6) for value in r] for r in U]}")

    return P, L, U, swaps


def forward_substitution(L, b):
    """Solves L z = b (L unit lower triangular), top row down."""
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
    A = [[1, 2, 3],
         [2, 4, 7],
         [3, 5, 3]]
    right_hand_sides = {"b1": [6, 13, 11],
                        "b2": [1, 0, 0],
                        "b3": [2, -1, 4]}
    n = 3

    print("=" * 70)
    print("(a) NAIVE DOOLITTLE -- WHERE IT BREAKS")
    print("=" * 70)
    try:
        naive_doolittle(A)
        print("  (unexpectedly succeeded)")
    except ZeroDivisionError as exc:
        print(f"  FAILED: {exc}")
        print("\n  Why: row 2 is exactly 2 * row 1 in the first TWO columns")
        print("       ([1,2,...] and [2,4,...]), so eliminating column 1 leaves")
        print("       U[2][2] = 4 - 2*2 = 0. Nothing is wrong with A -- it is")
        print("       non-singular. Naive Doolittle simply has no way to move a")
        print("       nonzero entry into that pivot slot.")

    print("\n" + "=" * 70)
    print("(b) PARTIAL PIVOTING:  P A = L U")
    print("=" * 70)
    P, L, U, swaps = plu_decompose(A)

    print(f"\n  P =")
    for row in P:
        print("   ", [int(value) for value in row])
    print(f"  L =")
    for row in L:
        print("   ", [round(value, 6) for value in row])
    print(f"  U =")
    for row in U:
        print("   ", [round(value, 6) for value in row])
    print(f"  swaps = {swaps}")

    PA = matmul(P, A)
    LU = matmul(L, U)
    diff = [[PA[i][j] - LU[i][j] for j in range(n)] for i in range(n)]
    manual_fro = frobenius_norm(diff)

    A_np = np.array(A, dtype=float)
    P_np, L_np, U_np = np.array(P), np.array(L), np.array(U)
    numpy_fro = float(np.linalg.norm(P_np @ A_np - L_np @ U_np, 'fro'))
    numpy_wrong = float(np.linalg.norm(A_np - L_np @ U_np, 'fro'))

    print(f"\n  manual ||PA - LU||_F = {manual_fro:.3e}   numpy = {numpy_fro:.3e}   <- ~0, correct")
    print(f"         ||A  - LU||_F = {numpy_wrong:.3e}"
          "   <- NOT ~0: it is PA that factors, not A")

    print("\n" + "=" * 70)
    print("(c) SOLVE ALL THREE RIGHT-HAND SIDES, REUSING THE SAME L AND U")
    print("=" * 70)
    solutions = {}
    for name, b in right_hand_sides.items():
        Pb = matvec(P, b)                       # <-- permute b the same way as A's rows
        z = forward_substitution(L, Pb)         # L z = P b
        x = backward_substitution(U, z)         # U x = z
        solutions[name] = x
        print(f"  {name} = {b}")
        print(f"      P @ {name} = {[round(value, 6) for value in Pb]}   "
              "<- forgetting this step is the classic silent bug")
        print(f"      z = {[round(value, 6) for value in z]}   (forward)")
        print(f"      x = {[round(value, 8) for value in x]}   (backward)")
    print("\n  L and U were computed ONCE and never touched again -- only the")
    print("  two O(n^2) triangular solves ran per right-hand side.")

    print("\n" + "=" * 70)
    print("(d) DETERMINANT FROM THE FACTORIZATION")
    print("=" * 70)
    diagonal = [U[i][i] for i in range(n)]
    det = (-1.0) ** swaps
    for value in diagonal:
        det *= value
    print(f"  det(P) = (-1)^{swaps} = {(-1) ** swaps},  det(L) = 1 (unit diagonal),")
    print(f"  det(U) = prod{[round(value, 6) for value in diagonal]} = "
          f"{np.prod(diagonal):.10f}")
    print(f"  det(A) = det(U) / det(P) = (-1)^{swaps} * det(U) = {det:.10f}")
    print(f"  np.linalg.det(A) = {np.linalg.det(A_np):.10f}")

    print("\n" + "=" * 70)
    print("(e) VERIFICATION")
    print("=" * 70)
    for name, b in right_hand_sides.items():
        x = solutions[name]
        b_np = np.array(b, dtype=float)
        Ax = matvec(A, x)
        residual = [Ax[i] - b[i] for i in range(n)]
        manual_res = l2_norm(residual)
        numpy_res = float(np.linalg.norm(A_np @ np.array(x) - b_np))
        x_lib = np.linalg.solve(A_np, b_np)
        print(f"  {name}: ours = {[round(value, 8) for value in x]}")
        print(f"       numpy = {np.round(x_lib, 8)}")
        print(f"       manual ||Ax-b||_2 = {manual_res:.3e}   numpy = {numpy_res:.3e}")
        assert np.allclose(x, x_lib, atol=1e-9)
        assert manual_res < 1e-9

    assert manual_fro < 1e-12
    assert numpy_wrong > 1e-3          # A itself really does NOT equal LU
    assert abs(det - np.linalg.det(A_np)) < 1e-8
    print("\n  ALL CHECKS PASSED")


# ----------------------------------------------------------------------
# WRITTEN QUESTION -- answer
#
# "Naive Doolittle failed even though A is non-singular. Explain the
#  difference between 'has no LU factorization' and 'has no solution', and
#  state what P is doing."
#
# These are unrelated properties, and it is worth being precise about it.
#
# "NO SOLUTION" is a statement about the SYSTEM A x = b: the equations
# contradict each other, which shows up as a row reducing to 0 = c with c != 0,
# and requires det(A) = 0. Here det(A) = 1 != 0, so A x = b has exactly one
# solution for every b -- the three solved above are proof.
#
# "NO LU FACTORIZATION (WITHOUT PIVOTING)" is a statement about the MATRIX and
# about the ROW ORDER it happens to be written in. Naive Doolittle needs every
# leading principal minor to be nonzero, because the k-th pivot is what it
# divides by at step k. Here the top-left 2x2 minor is
#       det([[1, 2], [2, 4]]) = 4 - 4 = 0
# because row 2 is exactly twice row 1 in those first two columns. Eliminating
# column 1 therefore leaves U[2][2] = 0, and with no pivoting there is nothing
# to divide by. The failure is caused by the ORDER the rows are written in, not
# by any defect in A -- and row order carries no mathematical information, since
# swapping two equations does not change the solution set.
#
# WHAT P DOES: P is a permutation matrix -- the identity with its rows
# reordered -- and P A is simply A with its equations rewritten in a better
# order. The theorem is that for ANY non-singular A there EXISTS a permutation
# P such that P A does have a plain LU factorization; partial pivoting is the
# constructive algorithm that finds one, by always choosing the largest
# available pivot. So the general statement is P A = L U, and A = L U is just
# the lucky special case P = I.
#
# THE PRACTICAL CONSEQUENCE, which is where marks are actually lost:
#   A x = b   ->   P A x = P b   ->   L U x = P b
# so the forward substitution must solve  L z = P b, NOT  L z = b. Skipping the
# permutation gives a plausible-looking x that solves nothing, with no error
# message. The self-check is ||A x - b||: if it comes out large while
# ||PA - LU|| is ~0, the factorization is fine and the missing P @ b is the bug.
# ----------------------------------------------------------------------
