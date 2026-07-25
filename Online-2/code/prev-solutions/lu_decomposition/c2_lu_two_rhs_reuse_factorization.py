"""
C2's online question (from res/Online Questions.txt, Q4):

  Given A (3x3) and two right-hand sides b1, b2 (3x1 each):
    1. Find the LU decomposition of A.
    2. Verify ||A - LU||.
    3. Solve both systems by forward + backward substitution, REUSING the
       already-computed L, U (do not re-factorize for b2).
    4. Calculate the L2 norm.
    5. Which part of the computation was reused, and why is it unnecessary
       to recompute L, U again?
    6. Verify both solutions with np.linalg.solve().
    7. Calculate the residual ||Ax - b|| for both.

The transcript doesn't record the exact A/b1/b2 used -- this uses a
representative diagonally-dominant 3x3 so plain (no-pivot) Doolittle LU is
stable. If the real exam matrix needs pivoting (a zero or tiny pivot shows
up), swap in `plu_decomposition` from algorithms/lu_decomposition.py and
verify ||P@A - L@U|| instead of ||A - L@U||.

Note on step 2 vs step 4: the transcript lists "verify ||A-LU||" and
"calculate L2 norm" as separate steps. Read together, that most likely means:
verify the factorization using TWO different matrix norms -- the Frobenius
norm (default, entrywise) and the L2/spectral norm (largest singular value,
via SVD). Both are computed below so either reading is covered.
"""
import numpy as np


def doolittle_lu(A, verbose=True):
    """Doolittle LU (no pivoting): A = L @ U, L unit-lower-triangular.
    Prints L, U after each column is eliminated."""
    A = A.astype(float).copy()
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]      # multiplier
            L[i, k] = m
            U[i, k:] -= m * U[k, k:]
        if verbose:
            print(f"\n--- after eliminating column {k} ---")
            print("L =\n", np.round(L, 4))
            print("U =\n", np.round(U, 4))

    return L, U


def forward_substitution(L, b):
    """Solves Lz = b (L unit lower triangular)."""
    n = len(b)
    z = np.zeros(n)
    for i in range(n):
        z[i] = (b[i] - L[i, :i] @ z[:i]) / L[i, i]
    return z


def backward_substitution(U, z):
    """Solves Ux = z (U upper triangular)."""
    n = len(z)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def solve_with_existing_lu(L, U, b):
    """Step 3: reuses an already-computed L, U -- no re-factorization."""
    z = forward_substitution(L, b)
    x = backward_substitution(U, z)
    return x


if __name__ == "__main__":
    A = np.array([[4, 2, 1],
                  [2, 5, 3],
                  [1, 3, 6]], dtype=float)
    b1 = np.array([7, 10, 10], dtype=float)
    b2 = np.array([1, -2, 3], dtype=float)

    # ---- 1. LU decomposition (once) ----
    print("=== Step 1: Doolittle LU decomposition of A ===")
    L, U = doolittle_lu(A)

    # ---- 2 & 4. Verify ||A - LU|| in two norms ----
    diff = A - L @ U
    fro_norm = np.linalg.norm(diff, 'fro')     # entrywise / Frobenius
    l2_norm = np.linalg.norm(diff, 2)          # spectral / induced 2-norm
    print("\n=== Step 2 & 4: verify factorization ===")
    print("||A - LU||_F  =", fro_norm)
    print("||A - LU||_2  =", l2_norm)

    # ---- 3. Solve BOTH systems, reusing the same L, U ----
    print("\n=== Step 3: solve Ax=b1 and Ax=b2, reusing L, U ===")
    x1 = solve_with_existing_lu(L, U, b1)
    x2 = solve_with_existing_lu(L, U, b2)
    print("x1 =", np.round(x1, 6))
    print("x2 =", np.round(x2, 6))

    # ---- 5. Written question: Reused computation explanation ----
    print("\n=== Step 5: Which part was reused and why? ===")
    print("Reused part: The LU factorization (matrices L and U).")
    print("Why unnecessary to recompute: L and U depend ONLY on matrix A, not on b.")
    print("Recomputing L, U for b2 would waste O(n^3) operations producing identical L and U.")
    print("Reusing L, U reduces the second solve to two cheap O(n^2) triangular solves.")

    # ---- 6. Verify against np.linalg.solve ----
    print("\n=== Step 6: verify vs np.linalg.solve ===")
    x1_np = np.linalg.solve(A, b1)
    x2_np = np.linalg.solve(A, b2)
    print("np.linalg.solve(A, b1) =", np.round(x1_np, 6))
    print("np.linalg.solve(A, b2) =", np.round(x2_np, 6))
    assert np.allclose(x1, x1_np) and np.allclose(x2, x2_np)

    # ---- 7. Residuals ----
    print("\n=== Step 7: residual ||Ax - b||_2 ===")
    print("||A@x1 - b1||_2 =", np.linalg.norm(A @ x1 - b1))
    print("||A@x2 - b2||_2 =", np.linalg.norm(A @ x2 - b2))


# ----------------------------------------------------------------------
# Written question (step 5):
# "Which part of the computation was reused, and why is it unnecessary to
#  calculate L, U again for the second right-hand side?"
#
# Answer:
#   The LU factorization (L and U themselves) is reused -- it is computed
#   exactly ONCE from A and never touched again. What changes between b1
#   and b2 is only the two O(n^2) triangular solves (forward substitution
#   Lz=b, backward substitution Ux=z); the O(n^3) elimination work that
#   built L and U is not repeated.
#
#   This is safe/correct because L and U depend ONLY on A, not on b. Since
#   A is the same matrix for both systems, its factorization PA=LU (or here
#   A=LU, no pivoting needed) is identical for b1 and b2 -- recomputing it
#   would do the exact same ~(2/3)n^3 flops of elimination a second time
#   and produce bit-for-bit the same L, U, purely wasted work. Reusing it
#   turns the second solve into two cheap O(n^2) triangular solves instead
#   of another full O(n^3) elimination -- this is precisely the "solve
#   against many right-hand sides" use case LU decomposition exists for
#   (see the written-question answer in lu_q1_doolittle_solve.py for the
#   general flop-count argument).
# ----------------------------------------------------------------------
