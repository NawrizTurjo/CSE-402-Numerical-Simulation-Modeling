"""
Complex case: rank deficiency > 1 (more than one free variable)
====================================================================
Past exam questions only ever needed ONE free variable (rank deficiency 1).
This demonstrates the general case: what if TWO rows collapse to zero
during elimination, not just one?

Example: 4 equations, but really only rank 2 worth of independent
information -- x1,x2 tied together, x3,x4 tied together, nothing pins down
the individual split. Two free variables needed.
"""
import numpy as np

EPS = 1e-9


def forward_elimination(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)
    for k in range(n - 1):
        p = k
        for i in range(k + 1, n):
            if abs(A[i, k]) > abs(A[p, k]):
                p = i
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[k], b[p] = b[p], b[k]
        if abs(A[k, k]) < EPS:
            continue
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]
    return A, b


def classify_and_solve(A, b, free_value=1.0, verbose=True):
    """Generalizes the single-free-variable case to ANY number of zero
    rows. See docs/gauss_elimination.md 'Code level' for why processing
    i=n-1 downto 0, assigning free_value at each zero row, is correct even
    with multiple simultaneous free variables."""
    U, c = forward_elimination(A, b)
    n = len(c)
    if verbose:
        print("augmented matrix after elimination:\n", np.round(np.column_stack([U, c]), 4))

    x = np.zeros(n)
    free_idx, inconsistent = [], []
    for i in range(n - 1, -1, -1):
        if np.all(np.abs(U[i]) < EPS):
            if abs(c[i]) > EPS:
                inconsistent.append(i)
                continue
            x[i] = free_value
            free_idx.append(i)
        else:
            x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

    if inconsistent:
        print(f"\n>>> NO SOLUTION -- rows {inconsistent} are contradictory (0 = nonzero)")
        return None
    if free_idx:
        print(f"\n>>> INFINITE SOLUTIONS -- {len(free_idx)} free variable(s) at index/indices {sorted(free_idx)}")
        print(f"    (each set to free_value={free_value} here; a real question may ask for a specific choice)")
    else:
        print("\n>>> UNIQUE SOLUTION")
    return x


if __name__ == "__main__":
    # rank 2 out of 4: rows 3,4 are exactly 2x rows 1,2 -- x1,x2 and x3,x4
    # are each individually free (only their COMBINATION is pinned down)
    A = np.array([[1, 1, 1, 1],
                  [2, 2, 2, 2],
                  [1, -1, 1, -1],
                  [2, -2, 2, -2]], dtype=float)
    b = np.array([4, 8, 0, 0], dtype=float)

    x = classify_and_solve(A, b)
    print("\nparticular solution x =", x)
    print("check ||Ax - b||_2 =", np.linalg.norm(A @ x - b))
    print("\nNOTE: rank(A) =", np.linalg.matrix_rank(A), "out of n =", len(b),
          "-- confirms 2 degrees of freedom, matching the 2 free variables found")
