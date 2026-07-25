"""
Inverse Power Method -- Canonical Library Functions & Theoretical Mappings
==========================================================================

THEORETICAL ALGORITHM MAPPING:
------------------------------
Computes the SMALLEST magnitude eigenvalue lambda_{min} and its eigenvector for matrix A.

1. SPECTRAL MAPPING THEOREM:
   If A * v = lambda * v, multiplying both sides by A^(-1) gives:
     A^(-1) * v = (1 / lambda) * v
   Therefore:
   - The eigenvalues of A^(-1) are mu_i = 1 / lambda_i.
   - The DOMINANT eigenvalue of A^(-1) (largest magnitude |mu|) corresponds to the
     SMALLEST magnitude eigenvalue of A (smallest |lambda|)!

2. SHIFTED INVERSE POWER METHOD:
   To find an eigenvalue near a specific scalar shift s:
     (A - s*I)^(-1) * v = mu * v   where mu = 1 / (lambda - s)
     => lambda = s + (1 / mu)

3. IMPLEMENTATION STRATEGIES:
   a) Explicit Matrix Inverse (inverse_power_naive):
      Computes Ainv = np.linalg.inv(A) once, then iterates y = Ainv @ x.
      Useful when exam questions explicitly instruct you to call np.linalg.inv().

   b) Factorization-based Triangular Solves (inverse_power_via_lu):
      NEVER computes matrix inverse A^(-1)!
      Instead:
      1. Factorize P * A = L * U ONCE before loop (O(n^3) work).
      2. In each iteration, solve (P * A) * y = P * x  <=>  L * (U * y) = P * x:
         - Forward sub:  L * z = P * x
         - Backward sub: U * y = z
      This reduces iteration cost to cheap O(n^2) triangular solves!
"""
import numpy as np


def inverse_power_naive(A, x0, tol=1e-8, max_iter=1000, hand_written=True, verbose=False):
    """
    Computes smallest magnitude eigenvalue using explicit np.linalg.inv(A).

    THEORETICAL STEPS:
    1. Form Ainv = A^(-1).
    2. Power iteration on Ainv: y = Ainv @ x.
    3. Dominant eigenvalue of Ainv is mu_est.
    4. Smallest eigenvalue of A is lambda_min = 1 / mu_est.
    """
    Ainv = np.linalg.inv(A)
    x = x0.astype(float).copy()
    history = []

    for it in range(max_iter):
        y = Ainv @ x
        if hand_written:
            idx = 0
            for j in range(1, len(y)):
                if abs(y[j]) > abs(y[idx]):
                    idx = j
        else:
            idx = np.argmax(np.abs(y))
        lam_est = y[idx]  # Eigenvalue estimate for A^(-1)
        history.append(lam_est)

        x_new = y / lam_est
        if verbose:
            print(f"  iter {it+1:3d}: eig(A^-1)={lam_est:10.6f} -> eig(A)={1/lam_est:10.6f}")

        if hand_written:
            max_diff = 0.0
            for j in range(len(x)):
                diff = abs(x_new[j] - x[j])
                if diff > max_diff:
                    max_diff = diff
        else:
            max_diff = np.max(np.abs(x_new - x))

        if max_diff < tol:
            x = x_new
            break

        x = x_new

    # Unit L2 normalization of eigenvector
    if hand_written:
        norm_sq = 0.0
        for val in x:
            norm_sq += val * val
        x = x / (norm_sq ** 0.5)
    else:
        x = x / np.linalg.norm(x)
    # Return lambda_min = 1.0 / mu, unit eigenvector, history
    return 1.0 / history[-1], x, history


def _plu(A, hand_written=True):
    """Helper: Partial pivoting LU factorization P*A = L*U."""
    A = A.astype(float).copy()
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        if hand_written:
            p = k
            for i in range(k + 1, n):
                if abs(U[i, k]) > abs(U[p, k]):
                    p = i
        else:
            p = k + int(np.argmax(np.abs(U[k:, k])))

        if p != k:
            if hand_written:
                for j in range(n):
                    U[k, j], U[p, j] = U[p, j], U[k, j]
                    P[k, j], P[p, j] = P[p, j], P[k, j]
                if k > 0:
                    for j in range(k):
                        L[k, j], L[p, j] = L[p, j], L[k, j]
            else:
                U[[k, p]] = U[[p, k]]
                P[[k, p]] = P[[p, k]]
                if k > 0:
                    L[[k, p], :k] = L[[p, k], :k]

        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            if hand_written:
                for j in range(k, n):
                    U[i, j] -= m * U[k, j]
            else:
                U[i, k:] -= m * U[k, k:]

    return P, L, U


def _fsub(L, b, hand_written=True):
    """Helper: Forward substitution L*z = b."""
    n = len(b)
    z = np.zeros(n)
    for i in range(n):
        if hand_written:
            total_sum = 0.0
            for j in range(i):
                total_sum += L[i, j] * z[j]
            z[i] = (b[i] - total_sum) / L[i, i]
        else:
            z[i] = (b[i] - L[i, :i] @ z[:i]) / L[i, i]
    return z


def _bsub(U, z, hand_written=True):
    """Helper: Backward substitution U*x = z."""
    n = len(z)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if hand_written:
            total_sum = 0.0
            for j in range(i + 1, n):
                total_sum += U[i, j] * x[j]
            x[i] = (z[i] - total_sum) / U[i, i]
        else:
            x[i] = (z[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def inverse_power_via_lu(A, x0, tol=1e-8, max_iter=500, hand_written=True, verbose=False):
    """
    Preferred LU-based Inverse Power Method: Avoids explicit inverse A^(-1).

    THEORETICAL STEPS:
    1. Factorize matrix A once into P * A = L * U.
    2. Iterate: Solve L * U * y = P * x using two cheap triangular solves.
    3. Extract dominant eigenvalue mu of A^(-1) and normalize vector x.
    4. Compute lambda_min(A) = 1 / mu.

    Returns:
        smallest_eigenvalue (float): lambda_min of A.
        unit_eigenvector (ndarray): Corresponding eigenvector (L2-norm = 1.0).
        history (list): Sequence of eigenvalue estimates of A^(-1).
    """
    # Factorize once: P * A = L * U
    P, L, U = _plu(A, hand_written=hand_written)
    x = x0.astype(float).copy()
    history = []

    for it in range(max_iter):
        # Step a: Forward substitution L * z = P * x
        z = _fsub(L, P @ x, hand_written=hand_written)

        # Step b: Backward substitution U * y = z
        y = _bsub(U, z, hand_written=hand_written)

        # Step c: Estimate eigenvalue mu for A^(-1)
        if hand_written:
            idx = 0
            for j in range(1, len(y)):
                if abs(y[j]) > abs(y[idx]):
                    idx = j
        else:
            idx = np.argmax(np.abs(y))
        lam_est = y[idx]
        history.append(lam_est)

        # Step d: Normalize vector x_new = y / mu
        x_new = y / lam_est

        if verbose:
            print(f"  iter {it+1:3d}: eig(A^-1)={lam_est:10.6f} -> eig(A)={1/lam_est:10.6f}")

        if hand_written:
            max_diff = 0.0
            for j in range(len(x)):
                diff = abs(x_new[j] - x[j])
                if diff > max_diff:
                    max_diff = diff
        else:
            max_diff = np.max(np.abs(x_new - x))

        if max_diff < tol:
            x = x_new
            break

        x = x_new

    # Unit L2 normalization of eigenvector
    if hand_written:
        norm_sq = 0.0
        for val in x:
            norm_sq += val * val
        x = x / (norm_sq ** 0.5)
    else:
        x = x / np.linalg.norm(x)
    return 1.0 / history[-1], x, history


def _self_test():
    """Self-test on A1 question matrix [[4,1,0,0],[1,3,1,0],[0,1,2,1],[0,0,1,1]]."""
    A = np.array([[4, 1, 0, 0], [1, 3, 1, 0], [0, 1, 2, 1], [0, 0, 1, 1]], dtype=float)
    x0 = np.array([1, 2, 3, -1], dtype=float)

    lam1, v1, _ = inverse_power_naive(A, x0)
    lam2, v2, _ = inverse_power_via_lu(A, x0)

    eigvals = np.linalg.eig(A)[0]
    lam_true = eigvals[np.argmin(np.abs(eigvals))]

    assert abs(lam1 - lam_true) < 1e-6 and abs(lam2 - lam_true) < 1e-6
    print("inverse_power_method.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
