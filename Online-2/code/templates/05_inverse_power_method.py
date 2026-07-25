"""
Inverse Power Method -- smallest-magnitude eigenvalue & eigenvector
======================================================================
Fact: if lambda is an eigenvalue of A, then 1/lambda is an eigenvalue of
A^-1. Running the (dominant) power method on A^-1 therefore finds the
SMALLEST eigenvalue of A -- take the reciprocal at the end.

Two versions below:
  1. `inverse_power_naive`      -- uses np.linalg.inv() explicitly (fine if
                                    the question allows/asks for it)
  2. `inverse_power_via_lu`     -- factor A = LU ONCE, then each iteration
                                    is two triangular solves instead of ever
                                    forming A^-1 (the "right" way -- see the
                                    written-question note at the bottom)
"""
import numpy as np


# ------------------------------------------------------------------
# 1. Naive version: np.linalg.inv() once, then plain power iteration
# ------------------------------------------------------------------
def inverse_power_naive(A, x0, tol=1e-8, max_iter=1000, verbose=True):
    Ainv = np.linalg.inv(A)
    x = x0.astype(float).copy()
    history = []

    for it in range(max_iter):
        y = Ainv @ x
        lam_est = y[np.argmax(np.abs(y))]     # dominant eigenvalue of A^-1
        history.append(lam_est)
        x_new = y / lam_est

        if verbose:
            print(f"  iter {it+1:3d}: eig(A^-1) est={lam_est:10.6f}  ->  eig(A) est={1/lam_est:10.6f}")

        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new

    x = x / np.linalg.norm(x)
    lam_min = 1.0 / history[-1]
    return lam_min, x, history


# ------------------------------------------------------------------
# 2. Preferred version: LU factor once, two triangular solves / iteration
# ------------------------------------------------------------------
def plu_decomposition(A):
    A = A.astype(float).copy()
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    for k in range(n - 1):
        p = k
        for i in range(k + 1, n):
            if abs(U[i, k]) > abs(U[p, k]):
                p = i
        if p != k:
            U[[k, p]] = U[[p, k]]
            P[[k, p]] = P[[p, k]]
            if k > 0:
                L[[k, p], :k] = L[[p, k], :k]
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            U[i, k:] -= m * U[k, k:]
    return P, L, U


def forward_substitution(L, b):
    n = len(b)
    z = np.zeros(n)
    for i in range(n):
        z[i] = (b[i] - L[i, :i] @ z[:i]) / L[i, i]
    return z


def backward_substitution(U, z):
    n = len(z)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def inverse_power_via_lu(A, x0, tol=1e-8, max_iter=500, verbose=True):
    P, L, U = plu_decomposition(A)     # factor ONCE, reuse every iteration
    x = x0.astype(float).copy()
    history = []

    for it in range(max_iter):
        Px = P @ x
        z = forward_substitution(L, Px)
        y = backward_substitution(U, z)      # y = A^-1 x, without ever forming A^-1
        lam_est = y[np.argmax(np.abs(y))]
        history.append(lam_est)
        x_new = y / lam_est

        if verbose:
            print(f"  iter {it+1:3d}: eig(A^-1) est={lam_est:10.6f}  ->  eig(A) est={1/lam_est:10.6f}")

        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new

    x = x / np.linalg.norm(x)
    lam_min = 1.0 / history[-1]
    return lam_min, x, history


if __name__ == "__main__":
    A = np.array([[4, 1, 0, 0],
                  [1, 3, 1, 0],
                  [0, 1, 2, 1],
                  [0, 0, 1, 1]], dtype=float)
    x0 = np.array([1, 2, 3, -1], dtype=float)   # avoid symmetric blind spots

    print("=== inverse power method (via LU, no explicit A^-1) ===")
    lam_min, v, history = inverse_power_via_lu(A, x0)
    print("\nsmallest eigenvalue :", lam_min)
    print("eigenvector         :", v)

    print("\n=== NumPy verification ===")
    eigvals, eigvecs = np.linalg.eig(A)
    k = np.argmin(np.abs(eigvals))
    lam_np, v_np = eigvals[k], eigvecs[:, k]
    print("smallest eigenvalue :", lam_np)
    print("eigenvector         :", v_np)

    if np.dot(v, v_np) < 0:
        v_np = -v_np
        print("(sign flipped for comparison)")

    print("\n|lambda diff| :", abs(lam_min - lam_np))
    print("||v diff||_2  :", np.linalg.norm(v - v_np))
    print("residual ||Av - lam*v||_2 :", np.linalg.norm(A @ v - lam_min * v))


# ----------------------------------------------------------------------
# Written-question quick answers
# ----------------------------------------------------------------------
# Q: Per iteration both "solve via LU" and "multiply by precomputed A^-1"
#    cost O(n^2) -- so why is the LU version still considered better?
# A: Setup cost: forming A^-1 costs ~2n^3 (n solves) vs ~2n^3/3 for a single
#    LU factorization -- 3x cheaper setup. Sparsity: if A is banded/sparse,
#    L and U preserve that structure (triangular solves stay cheap), but
#    A^-1 of a sparse matrix is generally DENSE -- for large sparse systems
#    this is the difference between feasible and infeasible.
