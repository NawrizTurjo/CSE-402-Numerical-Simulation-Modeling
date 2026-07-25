"""
Eigenvalue Decomposition (A = V * Lambda * V^(-1)) -- Canonical Library Functions & Theoretical Mappings
====================================================================================================

THEORETICAL ALGORITHM MAPPING:
------------------------------
Expresses a diagonalizable matrix A (n x n) in terms of its eigenvectors V and eigenvalues Lambda:
  A = V * Lambda * V^(-1)

1. MATRICES DEFINITION:
   - V = [v_1 | v_2 | ... | v_n]   (n x n matrix whose columns are eigenvectors v_i)
   - Lambda = diag(lambda_1, lambda_2, ..., lambda_n)   (n x n diagonal matrix of eigenvalues)
   - V^(-1) is the inverse of the eigenvector matrix V.

2. DERIVATION FROM EIGENVALUE EQUATION:
   By definition of eigenvalues and eigenvectors:
     A * v_i = lambda_i * v_i   for i = 1 ... n
   In matrix form:
     A * [v_1 | ... | v_n] = [v_1 | ... | v_n] * diag(lambda_1, ..., lambda_n)
     => A * V = V * Lambda
   Multiplying on the right by V^(-1):
     => A = V * Lambda * V^(-1)

3. MATRIX POWERS PROPERTY (Why Eigen-Decomposition is Useful):
   A^k = (V * Lambda * V^(-1)) * (V * Lambda * V^(-1)) ... (V * Lambda * V^(-1))
       = V * Lambda^k * V^(-1)
   Since Lambda is diagonal, Lambda^k = diag(lambda_1^k, ..., lambda_n^k), making matrix powers
   trivially cheap to compute in O(n) instead of O(k * n^3)!

4. CLOSED-FORM 2x2 MATRIX INVERSE FORMULA:
   For matrix M = [[a, b], [c, d]]:
     det(M) = a*d - b*c
     M^(-1) = (1 / det(M)) * [[d, -b], [-c, a]]
"""
import numpy as np


def inverse_2x2(M):
    """
    Computes closed-form inverse of a 2x2 matrix without library calls.

    THEORETICAL FORMULA:
    [[a, b], [c, d]]^(-1) = (1 / (a*d - b*c)) * [[d, -b], [-c, a]]
    """
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    det = a * d - b * c
    if abs(det) < 1e-12:
        raise ValueError("matrix is singular -- determinant is zero")
    return (1.0 / det) * np.array([[d, -b], [-c, a]])


def verify_via_numpy(A, hand_written=True):
    """
    Performs spectral decomposition verification A = V * Lambda * V^(-1).

    THEORETICAL STEPS:
    1. Obtain eigenvalues evals and eigenvectors evecs using np.linalg.eig(A)
       (no simple hand-written general eigensolver exists, so this call is
       library-only regardless of hand_written).
    2. Construct V = evecs and Lambda = diag(evals).
    3. Compute inverse V^(-1) (closed-form formula for 2x2 when hand_written).
    4. Reconstruct A_reconstructed = V @ Lambda @ V_inv.
    5. Evaluate Frobenius error norm ||A - A_reconstructed||_F.

    Returns:
        V (ndarray): Eigenvector matrix (columns are eigenvectors).
        Lambda (ndarray): Diagonal eigenvalue matrix.
        V_inv (ndarray): Inverse matrix V^(-1).
        A_reconstructed (ndarray): Re-synthesized matrix A.
        error (float): Frobenius norm of difference ||A - A_reconstructed||_F.
    """
    evals, evecs = np.linalg.eig(A)
    V = evecs
    n = V.shape[0]

    if hand_written:
        # Manual construction: place evals on the diagonal element-by-element
        Lambda = np.zeros((n, n), dtype=evals.dtype)
        for i in range(n):
            Lambda[i, i] = evals[i]
    else:
        Lambda = np.diag(evals)

    # Compute V^(-1) using closed-form 2x2 formula if applicable, or np.linalg.inv
    if hand_written and A.shape == (2, 2):
        V_inv = inverse_2x2(V.real)
    else:
        V_inv = np.linalg.inv(V)

    if hand_written:
        # Manual matrix multiply: A_reconstructed = (V @ Lambda) @ V_inv via triple loop
        VL = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                total = 0
                for k in range(n):
                    total += V[i, k] * Lambda[k, j]
                VL[i, j] = total

        A_reconstructed = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                total = 0
                for k in range(n):
                    total += VL[i, k] * V_inv[k, j]
                A_reconstructed[i, j] = total
        A_reconstructed = A_reconstructed.real

        # Manual Frobenius norm: sqrt(sum of squared differences)
        sum_sq = 0.0
        for i in range(n):
            for j in range(n):
                diff = A[i, j] - A_reconstructed[i, j]
                sum_sq += diff * diff
        error = sum_sq ** 0.5
    else:
        # Formula: A = V * Lambda * V^(-1)
        A_reconstructed = (V @ Lambda @ V_inv).real
        # Frobenius norm error check: ||A - A_reconstructed||_F
        error = np.linalg.norm(A - A_reconstructed, 'fro')

    return V, Lambda, V_inv, A_reconstructed, error


def _self_test():
    """Self-test on 2x2 (hand_written path) and 4x4 (vectorized path) matrices."""
    A = np.array([[2, 1], [1, 2]], dtype=float)
    *_, err = verify_via_numpy(A, hand_written=True)
    assert err < 1e-8, err

    A4 = np.array([[4, 1, 0, 0], [1, 5, 1, 0], [0, 1, 6, 1], [0, 0, 1, 7]], dtype=float)
    *_, err4 = verify_via_numpy(A4, hand_written=False)
    assert err4 < 1e-6, err4

    print("eigen_decomposition.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
