"""
C1's online question -- Power Iteration Method, dominant eigenpair
===================================================================

EXACT MATRIX, as recorded in res/A2_Prep.md section 5:

    A = [[12, 2, 1, 0],
         [ 2, 5, 0, 1],
         [ 1, 0, 3, 1],
         [ 0, 1, 1, 2]]

Steps the question asked for:
  1. Start from any nonzero x0.
  2. At each iteration compute y = A x_k.
  3. Normalize y by its largest-magnitude entry -- that entry IS the eigenvalue
     estimate, and the normalized vector is the eigenvector estimate.
  4. Repeat until the eigenvalue estimate converges.
  5. Verify using NumPy's eigenvalue computation.

Two notes given in the original question, both worth marks:
  - The eigenvector MUST be normalized before comparing to NumPy's, or they
    will not match numerically (we normalize by the largest entry, NumPy
    normalizes to unit Euclidean length -- different conventions).
  - NumPy's eigenvector may come back sign-flipped. v and -v are both valid
    eigenvectors, so that is EXPECTED, not an error.

Norms below are hand-computed first and cross-checked against NumPy after --
normalizing our own vector is part of our answer, not verification (see
res/A2_Prep.md section 1, and code/algorithms/manual_ops.py).
"""
import numpy as np


def l2_norm(vec):
    """Hand-written ||v||_2 = sqrt(sum of squares)."""
    total = 0.0
    for value in vec:
        total += value * value
    return total ** 0.5


def power_iteration(A, x0, tol=1e-6, max_iter=1000, verbose=True):
    """Power iteration: returns the dominant eigenvalue, the unit-normalized
    dominant eigenvector, and the list of eigenvalue estimates per iteration."""
    x = x0.astype(float).copy()
    eigen_estimates = []

    if verbose:
        print(f"{'iter':>4} | {'lambda estimate':>18} | x (largest entry = 1)")

    for _ in range(max_iter):
        y = A @ x
        # eigenvalue estimate = entry of largest MAGNITUDE (sign preserved).
        # np.max(y) would be wrong for a negative dominant eigenvalue.
        eigen_estimated = y[np.argmax(np.abs(y))]
        eigen_estimates.append(eigen_estimated)

        # scale x so its largest-magnitude entry is 1 (keeps iteration stable)
        x_new = y / eigen_estimated

        if verbose:
            print(f"{len(eigen_estimates):4d} | {eigen_estimated:18.10f} | "
                  f"{np.round(x_new, 6)}")

        # converge on the eigenVECTOR, not just the eigenvalue: the value
        # estimate can settle several iterations before the vector does
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new

    # normalize to unit 2-norm before returning, so it is comparable with
    # NumPy's eigenvector output -- computed by hand, checked against NumPy
    manual_norm = l2_norm(x)
    numpy_norm = float(np.linalg.norm(x))
    if verbose:
        print(f"\nmanual ||x|| = {manual_norm:.10f}   numpy ||x|| = {numpy_norm:.10f}")
    x = x / manual_norm
    return eigen_estimates[-1], x, eigen_estimates


# ----------------------------------------------------------------------
# The matrix given in the question
A = np.array([[12, 2, 1, 0],
              [2, 5, 0, 1],
              [1, 0, 3, 1],
              [0, 1, 1, 2]], dtype=float)
x0 = np.array([1, 1, 1, 1], dtype=float)

lam, v, history = power_iteration(A, x0)

print("\n=== Our power iteration ===")
print("dominant eigenvalue :", lam)
print("dominant eigenvector (unit length):", np.round(v, 8))
print("iterations          :", len(history))

# ----------------------------------------------------------------------
# Verification with NumPy
eigvals, eigvecs = np.linalg.eig(A)
k = np.argmax(np.abs(eigvals))          # index of the dominant eigenvalue
lam_np = eigvals[k].real
v_np = eigvecs[:, k].real               # eigenvectors are the COLUMNS

print("\n=== NumPy verification ===")
print("all eigenvalues     :", np.round(eigvals.real, 8))
print("dominant eigenvalue :", lam_np)
print("dominant eigenvector:", np.round(v_np, 8))

# NumPy may return the eigenvector with opposite sign: v and -v are both valid
# eigenvectors, so align the signs before comparing.
if np.dot(v, v_np) < 0:
    v_np = -v_np
    print("(sign flipped for comparison -- both v and -v are valid, "
          "this is not an error)")

# residual and differences hand-computed first, NumPy alongside as the check
Av = [sum(A[i, j] * v[j] for j in range(4)) for i in range(4)]
residual = [Av[i] - lam * v[i] for i in range(4)]
manual_res = l2_norm(residual)
numpy_res = float(np.linalg.norm(A @ v - lam * v))

vec_diff = l2_norm([v[i] - v_np[i] for i in range(4)])

print("\neigenvalue  difference     :", abs(lam - lam_np))
print("eigenvector difference     :", vec_diff)
print(f"residual ||Av - lam*v||_2  : manual = {manual_res:.3e}   "
      f"numpy = {numpy_res:.3e}")

# free global check: eigenvalues must sum to the trace
trace = sum(A[i, i] for i in range(4))
print(f"trace(A) = {trace}  vs  sum of numpy eigenvalues = {np.sum(eigvals.real):.8f}")

assert abs(lam - lam_np) < 1e-5 and vec_diff < 1e-5 and manual_res < 1e-5
print("\nALL CHECKS PASSED")


# ----------------------------------------------------------------------
# Notes worth remembering
#
# - Normalizing by the LARGEST ENTRY (not the Euclidean norm) is what makes
#   the divisor double as the eigenvalue estimate, and keeps the vector from
#   exploding as A^k x ~ lambda_1^k.
# - Our loop leaves the largest entry equal to 1; NumPy returns ||v||_2 = 1.
#   Convert to the same convention before comparing element-wise or the
#   difference will look enormous for a perfectly correct answer.
# - Convergence speed is set by |lambda_2 / lambda_1|. Here that is
#   4.80 / 12.64 ~ 0.38, giving clean convergence in well under 20 iterations.
#   A ratio near 1 would crawl; a ratio of exactly 1 (equal-magnitude or
#   complex-conjugate dominant pair) never converges at all -- see
#   code/complex_cases/.
# - The single most common bug in the collected code is using np.max(y)
#   instead of y[np.argmax(np.abs(y))]. They agree here because the dominant
#   eigenvalue is positive, and differ in both sign and value the moment it
#   is not.
# ----------------------------------------------------------------------
