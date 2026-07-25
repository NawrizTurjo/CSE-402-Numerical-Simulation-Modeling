"""
Hotelling's Deflation -- intermediate eigenvalues (2nd, 3rd, ...)
=====================================================================
The power method only ever finds the DOMINANT eigenvalue. To get the next
one, "mute" the dominant eigenvector's contribution and re-run power
iteration on the deflated matrix.

Steps:
  1. Power method on A -> lambda_1, v_1.
  2. Normalize v_1 to UNIT length: v_hat = v_1 / sqrt(v_1^T v_1).
  3. Deflate:  A_2 = A - lambda_1 * v_hat @ v_hat.T
  4. Power method on A_2 -> lambda_2, v_2. Repeat for A_3, A_4, ...

Requires A symmetric (eigenvectors orthogonal) for the "leaves v_2 untouched"
argument to hold cleanly -- see the written-question note below.

Caveat (explicitly called out in the theory slides): each deflation step
uses an approximate eigenvector, so errors compound. Reliable for the first
2-3 eigenvalues; degrades if you try to peel off too many.
"""
import numpy as np


def power_iteration(A, x0, tol=1e-8, max_iter=1000):
    x = x0.astype(float).copy()
    history = []
    for _ in range(max_iter):
        y = A @ x
        lam_est = y[np.argmax(np.abs(y))]
        history.append(lam_est)
        x_new = y / lam_est
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new
    x = x / np.linalg.norm(x)
    return history[-1], x, history


def deflate(A, lam, v):
    """A_next = A - lam * v_hat @ v_hat.T, v_hat = unit-normalized v."""
    v_hat = v / np.linalg.norm(v)
    return A - lam * np.outer(v_hat, v_hat)


def find_all_eigenpairs(A, x0, tol=1e-8, max_iter=1000, verbose=True):
    """Peels off eigenvalues one at a time via deflation.
    Returns (eigenvalues[], eigenvectors as columns of a matrix)."""
    n = A.shape[0]
    A_work = A.astype(float).copy()
    eigenvalues = []
    eigenvectors = []

    for i in range(n):
        lam, v, _ = power_iteration(A_work, x0, tol, max_iter)
        eigenvalues.append(lam)
        eigenvectors.append(v)
        if verbose:
            print(f"eigenpair {i+1}: lambda = {lam:.6f}   v = {np.round(v, 5)}")
        A_work = deflate(A_work, lam, v)

    return np.array(eigenvalues), np.column_stack(eigenvectors)


if __name__ == "__main__":
    # symmetric matrix from the slides: eigenvalues 10, 6, 4, 2
    A = np.array([[8, 2, 0, 0],
                  [2, 8, 0, 0],
                  [0, 0, 3, 1],
                  [0, 0, 1, 3]], dtype=float)
    # NOTE: x0=[1,1,1,1] is a trap here -- after deflating lambda_1=10, the
    # surviving eigenvector for lambda_2=6 is [1,-1,0,0]/sqrt(2), and [1,1,1,1]
    # has ZERO component along it (symmetric blind spot). Power iteration then
    # skips straight to lambda=4. Use a generic, non-symmetric x0 instead.
    x0 = np.array([1, 2, 3, -1], dtype=float)

    print("=== Step-by-step: find lambda_1, then deflate to find lambda_2 ===")
    lam1, v1, _ = power_iteration(A, x0)
    print("lambda_1 =", lam1, " v1 =", np.round(v1, 5))

    A2 = deflate(A, lam1, v1)
    print("\ndeflated matrix A2 =\n", np.round(A2, 4))

    lam2, v2, _ = power_iteration(A2, x0)
    print("\nlambda_2 =", lam2, " v2 =", np.round(v2, 5))

    print("\n=== All eigenpairs via repeated deflation ===")
    eigvals, V = find_all_eigenpairs(A, x0)
    print("\nour eigenvalues :", np.round(eigvals, 4))

    print("\n=== NumPy verification ===")
    np_vals, np_vecs = np.linalg.eig(A)
    order = np.argsort(-np.abs(np_vals))     # sort by |magnitude| descending, same as deflation order
    print("numpy eigenvalues (sorted):", np.round(np_vals[order], 4))


# ----------------------------------------------------------------------
# Written-question quick answers
# ----------------------------------------------------------------------
# Q: Why does deflation leave v_2 untouched when subtracting lambda_1's
#    contribution?
# A: A_2 = A - lam1 * v1_hat @ v1_hat.T. Applied to v2:
#    A_2 v2 = A v2 - lam1*v1_hat*(v1_hat . v2) = lam2*v2 - lam1*v1_hat*0
#    (v1_hat . v2 = 0 because eigenvectors of a SYMMETRIC matrix are
#    orthogonal) = lam2*v2. So v2 is still an eigenvector of A_2 with the
#    SAME eigenvalue lam2 -- nothing was disturbed.
# Q: Does deflation work for non-symmetric matrices?
# A: Not cleanly with this simple formula -- non-symmetric eigenvectors
#    aren't orthogonal in general, so subtracting lam1*v1_hat*v1_hat.T can
#    also perturb the other eigenvectors' directions.
# Q: Power iteration on the deflated matrix "converges" instantly to the
#    WRONG (smaller) eigenvalue -- what happened?
# A: Blind spot: your x0 has (numerically) zero component along the new
#    dominant eigenvector, usually because x0 is symmetric/aligned with a
#    symmetry the deflation just removed. Pick a generic x0 (e.g. distinct,
#    non-repeating entries) to avoid this -- same failure mode as the plain
#    power method's blind spot.
