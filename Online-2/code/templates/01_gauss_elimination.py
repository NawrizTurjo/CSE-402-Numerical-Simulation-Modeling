"""
Gauss Elimination with Partial Pivoting
========================================
Covers:
  - hand-written pivot selection + row swap (exam usually bans np.argmax/fancy indexing
    for this part -- toggle HAND_WRITTEN below)
  - forward elimination + back substitution
  - classifying Ax=b as unique / no solution / infinite solutions
  - determinant "for free" from the same elimination (see determinant() at bottom)

Copy the functions you need into your exam file and adapt matrix/vector literals.
"""
import numpy as np

EPS = 1e-9
HAND_WRITTEN = True   # set False to use np.argmax / fancy-index swaps instead


def forward_elimination(A, b, verbose=True):
    """Returns (U, c) -- upper-triangular A and transformed b.
    Prints every pivot chosen and every row swap performed."""
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    for k in range(n - 1):
        # ---- pivot selection: largest |entry| in column k, rows k..n-1 ----
        if HAND_WRITTEN:
            p = k
            for i in range(k + 1, n):
                if abs(A[i, k]) > abs(A[p, k]):
                    p = i
        else:
            p = k + int(np.argmax(np.abs(A[k:, k])))

        if verbose:
            print(f"  column {k}: pivot = {A[p, k]:.6g} (row {p})")

        # ---- row swap ----
        if p != k:
            if verbose:
                print(f"  row swap: R{k} <-> R{p}")
            if HAND_WRITTEN:
                for j in range(n):
                    A[k, j], A[p, j] = A[p, j], A[k, j]
            else:
                A[[k, p]] = A[[p, k]]
            b[k], b[p] = b[p], b[k]

        # entire column at/below k is zero -> nothing to eliminate this step
        if abs(A[k, k]) < EPS:
            if verbose:
                print(f"  pivot in column {k} is 0 -> skip elimination for this column")
            continue

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    return A, b


def back_substitution(U, c):
    n = len(c)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def classify_and_solve(A, b, name="System", free_value=5.0):
    """Full B1-style workflow: forward-eliminate, classify, solve/report.
    In the infinite-solutions case, hardcodes the LAST variable = free_value
    and back-solves the rest -- swap in whatever free-variable choice the
    question demands."""
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print("Augmented matrix before elimination:\n", np.column_stack([A, b]))

    U, c = forward_elimination(A, b)
    n = len(c)
    print("\nAugmented matrix after forward elimination:\n", np.round(np.column_stack([U, c]), 4))

    zero_rows = [i for i in range(n) if np.all(np.abs(U[i]) < EPS)]

    # ---------- NO SOLUTION:  0*x1 + 0*x2 + ... = c,  c != 0 ----------
    inconsistent = [i for i in zero_rows if abs(c[i]) > EPS]
    if inconsistent:
        print("\n>>> NO SOLUTION (inconsistent system)")
        for i in inconsistent:
            print(f"    row {i} reads:  0 = {c[i]:g}   (impossible)")
        return None

    # ---------- INFINITE SOLUTIONS:  0*x1 + 0*x2 + ... = 0 ----------
    if zero_rows:
        print("\n>>> INFINITE SOLUTIONS")
        for i in zero_rows:
            print(f"    row {i} reads:  0 = 0")
        x = np.zeros(n)
        x[n - 1] = free_value
        print(f"    choosing free variable x{n} = {free_value:g}")
        nonzero_rows = [i for i in range(n) if i not in zero_rows]
        for i in reversed(nonzero_rows):
            x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
        print("    one particular solution:", np.round(x, 4))
        print("    check ||Ax - b||_2 =", np.linalg.norm(A @ x - b))
        return x

    # ---------- UNIQUE SOLUTION ----------
    x = back_substitution(U, c)
    print("\n>>> UNIQUE SOLUTION")
    print("    x (our solver)  =", np.round(x, 6))
    print("    np.linalg.solve =", np.round(np.linalg.solve(A, b), 6))
    print("    ||Ax - b||_2    =", np.linalg.norm(A @ x - b))
    return x


def determinant(A, verbose=True):
    """det(A) via Gauss elimination with partial pivoting: (-1)^swaps * prod(diag(U))."""
    A = A.astype(float).copy()
    n = A.shape[0]
    swaps = 0
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(A[k:, k])))
        if p != k:
            A[[k, p]] = A[[p, k]]
            swaps += 1
        if abs(A[k, k]) < EPS:
            if verbose:
                print("Zero pivot -> determinant is 0")
            return 0.0
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
    det = ((-1) ** swaps) * np.prod(np.diag(A))
    if verbose:
        print(f"swaps={swaps}, diag(U)={np.round(np.diag(A), 4)}, det={det:.6g}")
    return det


if __name__ == "__main__":
    # System 1: unique solution
    A1 = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b1 = np.array([8, -11, -3], dtype=float)

    # System 2: no solution (row2 = 2*row1, inconsistent RHS)
    A2 = np.array([[1, 1, 1], [2, 2, 2], [1, 2, 3]], dtype=float)
    b2 = np.array([1, 3, 4], dtype=float)

    # System 3: infinite solutions (row2 = 2*row1, consistent RHS)
    A3 = np.array([[1, 1, 1], [2, 2, 2], [1, 2, 3]], dtype=float)
    b3 = np.array([6, 12, 10], dtype=float)

    classify_and_solve(A1, b1, "System 1 (unique)")
    classify_and_solve(A2, b2, "System 2 (no solution)")
    classify_and_solve(A3, b3, "System 3 (infinite solutions)")

    print(f"\n{'=' * 60}\nDeterminant demo\n{'=' * 60}")
    determinant(np.array([[0, 2, 1], [1, -2, -3], [-1, 1, 2]], dtype=float))


# ----------------------------------------------------------------------
# Written-question quick answers (see code/cheatsheet.md for the full bank)
# ----------------------------------------------------------------------
# Q: If a diagonal entry becomes 0, does that mean no solution?
# A: No. Two cases:
#    1. Some row BELOW still has a nonzero entry in that column -> pivoting
#       swaps it up, elimination continues, unique solution still possible.
#    2. The pivot AND everything below it in that column is also zero ->
#       matrix is singular. Then look at the RHS of the resulting zero row:
#         0 = c (c != 0)  -> NO SOLUTION (inconsistent)
#         0 = 0            -> INFINITE SOLUTIONS (free variable)
