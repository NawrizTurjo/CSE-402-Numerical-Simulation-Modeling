"""
Hotelling's Deflation -- Canonical Library Functions & Theoretical Mappings
=============================================================================

THEORETICAL ALGORITHM MAPPING:
------------------------------
Computes intermediate eigenvalues lambda_2, lambda_3, ..., lambda_n by "peeling off"
already computed eigenpairs from a SYMMETRIC matrix A.

1. HOTELLING DEFLATION FORMULA:
   Given dominant eigenpair (lambda_1, v_1) of matrix A:
     v_hat_1 = v_1 / ||v_1||_2           (Unit L2-normalized eigenvector)
     A_2 = A_1 - lambda_1 * (v_hat_1 * v_hat_1^T)

2. MATHEMATICAL PROOF OF DEFLATION (For Symmetric Matrix A):
   Symmetric matrix eigenvectors are mutually orthogonal:
     v_hat_i^T * v_hat_j = delta_{ij} = { 1 if i=j, 0 if i!=j }

   - Evaluating A_2 on v_hat_1:
       A_2 * v_hat_1 = (A_1 - lambda_1 * v_hat_1 * v_hat_1^T) * v_hat_1
                     = A_1 * v_hat_1 - lambda_1 * v_hat_1 * (v_hat_1^T * v_hat_1)
                     = lambda_1 * v_hat_1 - lambda_1 * v_hat_1 * (1) = 0
     -> The eigenvalue corresponding to v_hat_1 becomes 0!

   - Evaluating A_2 on remaining eigenvectors v_hat_j (j >= 2):
       A_2 * v_hat_j = (A_1 - lambda_1 * v_hat_1 * v_hat_1^T) * v_hat_j
                     = A_1 * v_hat_j - lambda_1 * v_hat_1 * (v_hat_1^T * v_hat_j)
                     = lambda_j * v_hat_j - lambda_1 * v_hat_1 * (0) = lambda_j * v_hat_j
     -> All other eigenpairs (lambda_j, v_j) remain completely unchanged!

3. REPEATED DEFLATION PROCEDURE:
   - For step i = 1 to n:
     1. Run power iteration on current working matrix A_work to find (lambda_i, v_i).
     2. Save (lambda_i, v_i).
     3. Update matrix: A_work <- A_work - lambda_i * (v_hat_i * v_hat_i^T).
"""
import numpy as np


def _power_iteration(A, x0, tol=1e-8, max_iter=1000, hand_written=True):
    """Internal Power Iteration helper to find dominant eigenpair of matrix A."""
    x = x0.astype(float).copy()
    history = []
    for _ in range(max_iter):
        y = A @ x
        if hand_written:
            idx = 0
            for j in range(1, len(y)):
                if abs(y[j]) > abs(y[idx]):
                    idx = j
        else:
            idx = np.argmax(np.abs(y))
        lam_est = y[idx]
        history.append(lam_est)
        x_new = y / lam_est

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

    if hand_written:
        norm_sq = 0.0
        for val in x:
            norm_sq += val * val
        x = x / (norm_sq ** 0.5)
    else:
        x = x / np.linalg.norm(x)
    return history[-1], x


def deflate(A, lam, v, hand_written=True):
    """
    Applies Hotelling's deflation formula to remove eigenpair (lam, v) from A.

    THEORETICAL FORMULA:
    A_next = A - lam * (v_hat @ v_hat.T)
    where v_hat = v / ||v||_2.

    Returns:
        A_next (ndarray): Deflated matrix with eigenvalue lam replaced by 0.
    """
    if hand_written:
        n = len(v)
        norm_sq = 0.0
        for val in v:
            norm_sq += val * val
        norm = norm_sq ** 0.5

        v_hat = np.zeros(n)
        for i in range(n):
            v_hat[i] = v[i] / norm

        A_next = A.copy()
        for i in range(n):
            for j in range(n):
                A_next[i, j] -= lam * v_hat[i] * v_hat[j]
        return A_next
    else:
        v_hat = v / np.linalg.norm(v)
        return A - lam * np.outer(v_hat, v_hat)


def find_all_eigenpairs(A, x0, tol=1e-8, max_iter=1000, hand_written=True, verbose=False):
    """
    Finds ALL eigenpairs of a symmetric matrix A via sequential power iteration & deflation.

    THEORETICAL PRECAUTION ON INITIAL GUESS x0:
    x0 must NOT be orthogonal to any target eigenvector of the deflated matrices.
    Using symmetric initial guesses (like [1, 1, 1, 1]) can cause zero components along
    intermediate eigenvectors. Always use a generic, asymmetric guess like [1, 2, 3, -1]!

    Returns:
        eigenvalues (ndarray): Array of computed eigenvalues [lambda_1, lambda_2, ...].
        eigenvectors (ndarray): Matrix whose columns are corresponding unit eigenvectors.
    """
    n = A.shape[0]
    A_work = A.astype(float).copy()
    eigenvalues, eigenvectors = [], []

    for i in range(n):
        # Step 1: Power iteration to extract dominant eigenpair of A_work
        lam, v = _power_iteration(A_work, x0, tol, max_iter, hand_written=hand_written)
        eigenvalues.append(lam)
        eigenvectors.append(v)

        if verbose:
            print(f"eigenpair {i+1}: lambda={lam:.6f}  v={np.round(v,5)}")

        # Step 2: Deflate matrix A_work to set lambda to 0 for next iteration
        A_work = deflate(A_work, lam, v, hand_written=hand_written)

    return np.array(eigenvalues), np.column_stack(eigenvectors)


def _self_test():
    """Self-test deflation on 4x4 matrix with true eigenvalues [10, 6, 4, 2]."""
    A = np.array([[8, 2, 0, 0], [2, 8, 0, 0], [0, 0, 3, 1], [0, 0, 1, 3]], dtype=float)
    x0 = np.array([1, 2, 3, -1], dtype=float)  # Generic asymmetric guess
    eigvals, _ = find_all_eigenpairs(A, x0)
    assert np.allclose(sorted(eigvals, reverse=True), [10, 6, 4, 2], atol=1e-6), eigvals
    print("deflation.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
