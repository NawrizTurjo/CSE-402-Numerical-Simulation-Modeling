"""
C2's online question -- LU decomposition reused for two right-hand sides
=========================================================================

EXACT DATA, as recorded in res/A2_Prep.md section 6:

    A  = [[4, 3, 2],        b1 = [1, 2, 3]        b2 = [4, 5, 6]
          [2, 5, 3],
          [1, 2, 4]]

The question, verbatim in structure:
  1. Find the LU decomposition of A.
  2. Verify the decomposition by checking ||A - LU||.
  3. Solve A x = b1 AND A x = b2 via forward then back substitution,
     REUSING the same L, U for both.
  4. Calculate the L2 norm (of the A - LU reconstruction error from step 2,
     as a single number).
  5. Which part of the computation was reused, and why is it unnecessary to
     recompute L, U for the second right-hand side?
  6. Verify both solutions against np.linalg.solve().
  7. Calculate the residual ||Ax - b|| for both solutions.

This A is diagonally dominant enough that plain (no-pivot) Doolittle is stable
and exact -- no pivoting needed, ||A - LU|| comes out at 0. If a variant of
this question hands you a matrix that DOES need pivoting (a zero or tiny pivot
appears), switch to `plu_decomposition` from algorithms/lu_decomposition.py,
verify ||P A - L U|| instead of ||A - LU||, and solve L z = P b rather than
L z = b. See practice/extra/e3_plu_when_naive_lu_fails.py for that case worked
end to end.

Every norm and residual below is HAND-COMPUTED first and cross-checked against
NumPy after -- they are part of the answer, not verification (res/A2_Prep.md
section 1). Both numbers are printed and must agree.
"""
import numpy as np


# ----------------------------------------------------------------------
# hand-written primitives -- ||A-LU|| and ||Ax-b|| are OUR results
# ----------------------------------------------------------------------

def l2_norm(vec):
    """||v||_2 = sqrt(sum of squares)."""
    total = 0.0
    for value in vec:
        total += value * value
    return total ** 0.5


def matvec(M, v):
    """M @ v, row by row."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def matmul(X, Y):
    """X @ Y, triple loop."""
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
    """||M||_F -- flatten and take the ordinary L2 norm of all entries."""
    total = 0.0
    for row in M:
        for value in row:
            total += value * value
    return total ** 0.5


# ----------------------------------------------------------------------
# the method itself
# ----------------------------------------------------------------------

def doolittle_lu(A, verbose=True):
    """
    Doolittle LU (no pivoting): A = L U, with L unit lower triangular.

    Computes U row by row and L column by column, which is the form the
    lecture derives -- each entry is "the original entry minus what the
    already-known factors account for":
        U[i][j] = A[i][j] - sum_{k<i} L[i][k] U[k][j]     (j >= i)
        L[j][i] = (A[j][i] - sum_{k<i} L[j][k] U[k][i]) / U[i][i]   (j > i)
    """
    n = len(A)
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i, n):                      # row i of U
            total = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - total
            if verbose:
                print(f"  U[{i + 1}][{j + 1}] = {A[i][j]} - {total:g} = {U[i][j]:g}")
        for j in range(i + 1, n):                  # column i of L, below diagonal
            total = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - total) / U[i][i]
            if verbose:
                print(f"  L[{j + 1}][{i + 1}] = ({A[j][i]} - {total:g}) / "
                      f"{U[i][i]:g} = {L[j][i]:g}")

    return L, U


def forward_substitution(L, b, verbose=True):
    """Solves L z = b, top row down (L unit lower triangular, L[i][i] = 1)."""
    n = len(b)
    z = [0.0] * n
    for i in range(n):
        total = b[i]
        for j in range(i):
            total -= L[i][j] * z[j]
        z[i] = total / L[i][i]
        if verbose:
            print(f"    z{i + 1} = {z[i]:g}")
    return z


def backward_substitution(U, z, verbose=True):
    """Solves U x = z, bottom row up."""
    n = len(z)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = z[i]
        for j in range(i + 1, n):
            total -= U[i][j] * x[j]
        x[i] = total / U[i][i]
        if verbose:
            print(f"    x{i + 1} = {x[i]:g}")
    return x


def solve_with_existing_lu(L, U, b, verbose=True):
    """Step 3: reuses an ALREADY-COMPUTED L, U -- no re-factorization here."""
    z = forward_substitution(L, b, verbose=verbose)
    x = backward_substitution(U, z, verbose=verbose)
    return x, z


if __name__ == "__main__":
    A = [[4, 3, 2],
         [2, 5, 3],
         [1, 2, 4]]
    b1 = [1, 2, 3]
    b2 = [4, 5, 6]
    n = 3

    # ---- 1. LU decomposition, computed exactly ONCE ----
    print("=" * 70)
    print("1. LU DECOMPOSITION (Doolittle)")
    print("=" * 70)
    L, U = doolittle_lu(A)
    print(f"\n  L = {[[round(value, 6) for value in row] for row in L]}")
    print(f"  U = {[[round(value, 6) for value in row] for row in U]}")

    # ---- 2 & 4. Verify ||A - LU|| ----
    print("\n" + "=" * 70)
    print("2 & 4. VERIFY THE DECOMPOSITION:  ||A - LU||")
    print("=" * 70)
    LU = matmul(L, U)
    diff = [[A[i][j] - LU[i][j] for j in range(n)] for i in range(n)]
    manual_norm = frobenius_norm(diff)

    A_np = np.array(A, dtype=float)
    L_np, U_np = np.array(L), np.array(U)
    numpy_norm = float(np.linalg.norm(A_np - L_np @ U_np))
    numpy_spectral = float(np.linalg.norm(A_np - L_np @ U_np, 2))

    print(f"  L @ U = {[[round(value, 6) for value in row] for row in LU]}")
    print(f"  A     = {A}")
    print(f"\n  manual ||A - LU||_2 (entrywise/Frobenius) = {manual_norm:.3e}")
    print(f"  numpy  ||A - LU||   (same thing)           = {numpy_norm:.3e}")
    print(f"  numpy  ||A - LU||_2 (spectral, largest singular value) = "
          f"{numpy_spectral:.3e}")
    print("  -> exact reconstruction; no pivoting was needed for this A")
    print("\n  (Step 4's 'L2 norm' is the same single number as step 2 -- the")
    print("   Frobenius norm of the difference matrix. The spectral 2-norm is")
    print("   printed too so either reading of the question is covered.)")

    # ---- 3. Solve BOTH systems, reusing the same L and U ----
    print("\n" + "=" * 70)
    print("3. SOLVE, REUSING THE SAME L AND U")
    print("=" * 70)
    solutions = {}
    for name, b in [("b1", b1), ("b2", b2)]:
        print(f"\n  --- {name} = {b} ---")
        print("  forward substitution L z = b:")
        x, z = solve_with_existing_lu(L, U, b)
        solutions[name] = x
        print(f"  solution x = {[round(value, 6) for value in x]}")
    print("\n  L and U were NOT recomputed between b1 and b2 -- only the")
    print("  forward-substitution input changed.")

    # ---- 6. Verify against np.linalg.solve ----
    print("\n" + "=" * 70)
    print("6. VERIFY WITH np.linalg.solve()")
    print("=" * 70)
    for name, b in [("b1", b1), ("b2", b2)]:
        x_np = np.linalg.solve(A_np, np.array(b, dtype=float))
        print(f"  {name}: ours  = {[round(value, 8) for value in solutions[name]]}")
        print(f"      numpy = {np.round(x_np, 8)}")
        assert np.allclose(solutions[name], x_np, atol=1e-10)

    # ---- 7. Residuals ----
    print("\n" + "=" * 70)
    print("7. RESIDUAL  r = Ax - b  FOR BOTH")
    print("=" * 70)
    for name, b in [("b1", b1), ("b2", b2)]:
        x = solutions[name]
        Ax = matvec(A, x)                                   # manual A @ x
        residual = [Ax[i] - b[i] for i in range(n)]         # manual subtraction
        manual_res = l2_norm(residual)                      # manual L2 norm
        numpy_res = float(np.linalg.norm(
            A_np @ np.array(x) - np.array(b, dtype=float)))
        print(f"  {name}: Ax - b = {[round(value, 12) for value in residual]}")
        print(f"       manual ||.||_2 = {manual_res:.3e}   numpy = {numpy_res:.3e}")
        assert manual_res < 1e-10

    assert manual_norm < 1e-12
    print("\n  ALL CHECKS PASSED")


# ----------------------------------------------------------------------
# 5. WRITTEN QUESTION -- answer
#
# "Which part of the computation was reused, and why is it unnecessary to
#  calculate L, U again for the second right-hand side?"
#
# WHAT WAS REUSED: the decomposition itself -- L and U. They are computed
# exactly ONCE from A and never touched again. Only the two triangular solves
# (forward substitution L z = b, backward substitution U x = z) are redone per
# right-hand side.
#
# WHY THAT IS VALID: L and U are properties of A ALONE. Look at the Doolittle
# formulas -- every entry of L and U is built from entries of A and from
# previously-computed entries of L and U. The vector b appears nowhere in the
# derivation; it only enters afterwards, as the input to L z = b. Changing b
# therefore cannot change the factorization. Recomputing it for b2 would run
# the identical arithmetic and produce bit-for-bit the same L and U.
#
# WHY IT MATTERS -- the cost:
#   LU factorization:            ~(2/3) n^3 flops, paid ONCE
#   forward + back substitution: ~2 n^2 flops, paid per right-hand side
#
# So for m right-hand sides:
#   LU, reused:                  O(n^3) + m * O(n^2)
#   Gauss elimination per RHS:   m * O(n^3)
#
# At m = 2 and n = 3 the saving is small. At n = 100 and m = 50 it is the
# difference between ~33 million flops and ~1.7 billion -- roughly 50x, from
# the same arithmetic organized once instead of fifty times. This is the entire
# reason LU exists as a separate method rather than being folded into Gauss
# elimination: elimination on the augmented [A|b] entangles the work with that
# particular b and leaves nothing reusable, whereas LU keeps the expensive part
# separate from the cheap part.
#
# The same argument, one level up, is why the matrix inverse is computed by
# solving A x_j = e_j for the n columns of the identity off ONE factorization
# (practice/a2prep/p1b_matrix_inverse_gj_and_lu.py) -- the identity's columns
# are just n right-hand sides.
# ----------------------------------------------------------------------
