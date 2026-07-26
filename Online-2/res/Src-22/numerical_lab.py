"""
CSE401 Lab Prep — Reference Implementations (all work for any n x n system)
---------------------------------------------------------------------------
1. Gauss Elimination (with partial pivoting)
2. Gauss-Jordan Elimination (solve + matrix inverse)
3. LU Decomposition (Doolittle, L has 1s on diagonal) + solve
4. Power Method (dominant eigenvalue/eigenvector)
5. Deflation (second eigenvalue)

Each algorithm is written from scratch (only numpy arrays used as containers),
then verified against numpy's built-in solvers at the bottom.
"""

import numpy as np


# ----------------------------------------------------------------------
# 1. GAUSS ELIMINATION with partial pivoting
# ----------------------------------------------------------------------
def gauss_elimination(A, b):
    """Solve Ax = b. Forward elimination -> upper triangular -> back substitution."""
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    # ---- Forward elimination ----
    for k in range(n - 1):                      # pivot column k
        # partial pivoting: swap in the row with the largest |value| in column k
        p = k + np.argmax(np.abs(A[k:, k]))
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]

        for i in range(k + 1, n):               # rows below the pivot
            m = A[i, k] / A[k, k]               # multiplier
            A[i, k:] -= m * A[k, k:]            # R_i <- R_i - m * R_k
            b[i]     -= m * b[k]

    # ---- Back substitution (bottom to top) ----
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i]
    return x


# ----------------------------------------------------------------------
# 2. GAUSS-JORDAN ELIMINATION  (reduce A to identity)
# ----------------------------------------------------------------------
def gauss_jordan(A, b):
    """Solve Ax = b by reducing [A|b] to [I|x]. No back substitution needed."""
    n = len(b)
    aug = np.hstack([A.astype(float), b.astype(float).reshape(-1, 1)])

    for k in range(n):
        # partial pivoting
        p = k + np.argmax(np.abs(aug[k:, k]))
        if p != k:
            aug[[k, p]] = aug[[p, k]]

        aug[k] /= aug[k, k]                     # normalize pivot row -> pivot = 1
        for i in range(n):                      # clear column k in ALL other rows
            if i != k:
                aug[i] -= aug[i, k] * aug[k]

    return aug[:, -1]                           # last column is the solution


def gauss_jordan_inverse(A):
    """Invert A by reducing [A|I] to [I|A^-1]."""
    n = A.shape[0]
    aug = np.hstack([A.astype(float), np.eye(n)])

    for k in range(n):
        p = k + np.argmax(np.abs(aug[k:, k]))
        if p != k:
            aug[[k, p]] = aug[[p, k]]
        aug[k] /= aug[k, k]
        for i in range(n):
            if i != k:
                aug[i] -= aug[i, k] * aug[k]

    return aug[:, n:]                           # right half is A^-1


# ----------------------------------------------------------------------
# 3. LU DECOMPOSITION (Doolittle: A = L U, ones on L's diagonal)
#    NOTE: no pivoting here, matching the standard theory-class version.
# ----------------------------------------------------------------------
def lu_decompose(A):
    """Return L, U with A = L @ U. L stores the elimination multipliers."""
    A = A.astype(float)
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]               # multiplier m_ik
            L[i, k] = m                         # store it in L
            U[i, k:] -= m * U[k, k:]            # eliminate as in Gauss
    return L, U


def lu_solve(L, U, b):
    """Solve Ax=b given A=LU:  Ly=b (forward sub), then Ux=y (back sub)."""
    n = len(b)
    # forward substitution (top to bottom), L diagonal is all 1s
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - L[i, :i] @ y[:i]
    # back substitution (bottom to top)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x


# ----------------------------------------------------------------------
# 4. POWER METHOD (dominant eigenvalue + eigenvector)
# ----------------------------------------------------------------------
def power_method(A, tol=1e-10, max_iter=1000):
    """Iterate x <- Ax, normalize each step. Returns (lambda_1, v_1)."""
    n = A.shape[0]
    x = np.ones(n)                              # any nonzero starting vector
    x /= np.linalg.norm(x)

    for _ in range(max_iter):
        y = A @ x
        x_new = y / np.linalg.norm(y)           # normalize (unit length)
        if x_new @ x < 0:                       # fix sign flip (negative lambda)
            x_new = -x_new
        if np.linalg.norm(x_new - x) < tol:     # converge on the VECTOR
            x = x_new
            break
        x = x_new

    lam = x @ (A @ x) / (x @ x)                 # Rayleigh quotient at the end
    return lam, x


# ----------------------------------------------------------------------
# 5. DEFLATION (find the 2nd eigenvalue after the 1st)
#    A2 = A - lambda_1 * v1 v1^T   (v1 must be a UNIT vector; A symmetric)
# ----------------------------------------------------------------------
def deflate(A, lam1, v1):
    v1 = v1 / np.linalg.norm(v1)
    return A - lam1 * np.outer(v1, v1)


def second_eigenvalue(A):
    lam1, v1 = power_method(A)
    A2 = deflate(A, lam1, v1)
    lam2, v2 = power_method(A2)
    return lam1, v1, lam2, v2


# ======================================================================
# VERIFICATION — run this file to check everything against numpy
# ======================================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(42)

    # --- random n x n test system (change n freely) ---
    n = 5
    A = rng.uniform(-10, 10, (n, n))
    A += n * np.eye(n)                          # make it well-conditioned
    b = rng.uniform(-10, 10, n)

    x_true = np.linalg.solve(A, b)              # numpy's answer = ground truth

    print(f"Testing on a random {n}x{n} system\n" + "=" * 45)

    # 1. Gauss elimination
    x = gauss_elimination(A, b)
    print("Gauss elimination   ok?", np.allclose(x, x_true))

    # 2. Gauss-Jordan
    x = gauss_jordan(A, b)
    print("Gauss-Jordan        ok?", np.allclose(x, x_true))

    # 2b. Gauss-Jordan inverse
    Ainv = gauss_jordan_inverse(A)
    print("GJ inverse          ok?", np.allclose(Ainv, np.linalg.inv(A)))
    print("   check A @ A^-1 = I:", np.allclose(A @ Ainv, np.eye(n)))

    # 3. LU decomposition
    L, U = lu_decompose(A)
    print("LU: A = L@U         ok?", np.allclose(L @ U, A))
    print("   L lower-tri, U upper-tri:",
          np.allclose(L, np.tril(L)), np.allclose(U, np.triu(U)))
    x = lu_solve(L, U, b)
    print("LU solve            ok?", np.allclose(x, x_true))

    # 4 & 5. Power method + deflation (use a SYMMETRIC matrix so
    #        eigenvectors are orthogonal and deflation is valid)
    S = rng.uniform(-5, 5, (n, n))
    S = (S + S.T) / 2                           # symmetrize
    lam1, v1, lam2, v2 = second_eigenvalue(S)

    true_vals = np.linalg.eigvalsh(S)           # numpy ground truth
    order = np.argsort(np.abs(true_vals))[::-1] # sort by |lambda|
    t1, t2 = true_vals[order[0]], true_vals[order[1]]

    print(f"Power method lam1 = {lam1:.6f}   numpy = {t1:.6f}   ok?",
          np.isclose(lam1, t1))
    print(f"Deflation    lam2 = {lam2:.6f}   numpy = {t2:.6f}   ok?",
          np.isclose(lam2, t2))
    # eigenvector check: A v = lambda v
    print("Eigvec check A@v1 = lam1*v1 ok?",
          np.allclose(S @ v1, lam1 * v1, atol=1e-6))

    # --- a small hand-checkable example (good for exam practice) ---
    print("\nHand-checkable 3x3 example\n" + "=" * 45)
    A3 = np.array([[2., 1., -1.],
                   [-3., -1., 2.],
                   [-2., 1., 2.]])
    b3 = np.array([8., -11., -3.])
    print("A =\n", A3)
    print("b =", b3)
    print("Gauss solution:", gauss_elimination(A3, b3), "(expected [2, 3, -1])")
    L, U = lu_decompose(A3)
    print("L =\n", L, "\nU =\n", U)
    print("LU solve      :", lu_solve(L, U, b3))
