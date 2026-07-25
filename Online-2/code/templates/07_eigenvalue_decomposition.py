"""
Eigenvalue Decomposition -- A = V * Lambda * V^-1
=====================================================
V   = matrix whose COLUMNS are the eigenvectors of A.
Lambda = diagonal matrix of the corresponding eigenvalues (same order as V's columns).
V^-1 translates standard coords -> eigen-coords; V translates back.

Two flavors below:
  1. `verify_via_numpy`   -- get eigenpairs from np.linalg.eig, assemble
                              V, Lambda, V^-1, reconstruct A, check the error.
  2. `inverse_2x2`        -- explicit 2x2 inverse formula, for when the
                              question specifically bans np.linalg.inv().
"""
import numpy as np


def inverse_2x2(M):
    """Explicit closed-form inverse of a 2x2 matrix [[a,b],[c,d]]."""
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    det = a * d - b * c
    if abs(det) < 1e-12:
        raise ValueError("matrix is singular")
    return (1.0 / det) * np.array([[d, -b], [-c, a]])


def verify_via_numpy(A, use_hand_inverse_2x2=False):
    evals, evecs = np.linalg.eig(A)
    V = evecs                       # columns are eigenvectors
    Lambda = np.diag(evals)

    if use_hand_inverse_2x2 and A.shape == (2, 2):
        V_inv = inverse_2x2(V.real)
    else:
        V_inv = np.linalg.inv(V)

    A_reconstructed = V @ Lambda @ V_inv

    print("eigenvalues:", np.round(evals.real, 6))
    print("V (columns = eigenvectors):\n", np.round(V.real, 6))
    print("Lambda:\n", np.round(Lambda.real, 6))
    print("V^-1:\n", np.round(V_inv.real, 6))
    print("reconstructed A = V @ Lambda @ V^-1:\n", np.round(A_reconstructed.real, 6))
    print("original A:\n", A)

    err = np.linalg.norm(A - A_reconstructed.real, 'fro')
    print(f"||A - V@Lambda@V^-1||_F = {err:.3e}  ->  {'OK' if err < 1e-8 else 'CHECK'}")
    return V, Lambda, V_inv, A_reconstructed.real


if __name__ == "__main__":
    print("=" * 60, "\n2x2 example (hand inverse formula)\n", "=" * 60)
    A2 = np.array([[2, 1], [1, 2]], dtype=float)
    verify_via_numpy(A2, use_hand_inverse_2x2=True)

    print("\n", "=" * 60, "\n4x4 example (np.linalg.inv)\n", "=" * 60)
    A4 = np.array([[4, 1, 0, 0],
                   [1, 5, 1, 0],
                   [0, 1, 6, 1],
                   [0, 0, 1, 7]], dtype=float)
    verify_via_numpy(A4)


# ----------------------------------------------------------------------
# Written-question quick answers
# ----------------------------------------------------------------------
# Q: Why is Lambda diagonal?
# A: Each eigenvector direction is scaled INDEPENDENTLY by its own
#    eigenvalue (Av_i = lam_i v_i, no mixing between directions) -- that
#    independence is exactly what "diagonal" encodes.
# Q: Why is A = V Lambda V^-1 useful for computing A^k?
# A: A^k = V Lambda^k V^-1 -- the middle V^-1 V pairs cancel telescopically
#    for every intermediate power, so you translate ONCE at the start and
#    ONCE at the end; Lambda^k is trivial (diagonal, just raise each entry).
# Q: Derivation, one line?
# A: Av_i = lam_i v_i for every i  =>  A[v1 ... vn] = [lam1 v1 ... lamn vn]
#    = [v1...vn] diag(lam1..lamn)  =>  A V = V Lambda  =>  A = V Lambda V^-1.
