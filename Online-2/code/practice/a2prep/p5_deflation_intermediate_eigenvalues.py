"""
P5 -- Deflation for Intermediate Eigenvalues     [~30 min]  source: res/A2_Prep.md sec. 12
==========================================================================================

QUESTION
--------
Using C1's matrix
    A = [[12, 2, 1, 0], [2, 5, 0, 1], [1, 0, 3, 1], [0, 1, 1, 2]]

(a) Run the power method on A to get the dominant eigenpair (lambda1, v1),
    with v1 normalized to unit length.
(b) Build the DEFLATED matrix  A2 = A - lambda1 * v1 v1^T  (outer product of
    the unit eigenvector with itself, scaled by lambda1).
(c) Run the power method AGAIN on A2 to get (lambda2, v2) -- the SECOND-largest
    eigenvalue of the original A.
(d) Verify A2 v1 ~= 0 (lambda1 has been muted to 0 in A2's spectrum) and
    v1 . v2 ~= 0 (the two eigenvectors are orthogonal).

WRITTEN QUESTION: this trick relies on A being symmetric. Why -- what
specifically would break if A weren't?

WHAT THIS DRILLS
----------------
The highest-risk topic on the syllabus: last one taught, never yet tested, and
built entirely on the power method (which HAS been tested). Deflation is not a
new algorithm -- it is "run the power method again, on a modified matrix."

WHY IT WORKS:
  A2 = A - lambda1 * v1_hat v1_hat^T, with ||v1_hat|| = 1.
  Apply A2 to v1_hat:
      A2 v1_hat = A v1_hat - lambda1 v1_hat (v1_hat . v1_hat)
                = lambda1 v1_hat - lambda1 v1_hat * 1
                = 0                              <- lambda1 becomes 0
  Apply A2 to any OTHER eigenvector v_j:
      A2 v_j = A v_j - lambda1 v1_hat (v1_hat . v_j)
             = lambda_j v_j - lambda1 v1_hat * 0     [orthogonality!]
             = lambda_j v_j                     <- untouched
  So A2 has exactly A's spectrum with lambda1 replaced by 0. The power method
  on A2 therefore finds the next-largest eigenvalue of the ORIGINAL A.
  That second line is where symmetry is load-bearing.
"""
import numpy as np


def l2_norm(vec):
    """Hand-written ||v||_2 -- normalizing our own vector is part of the answer."""
    total = 0.0
    for value in vec:
        total += value * value
    return total ** 0.5


def matvec(M, v):
    """Hand-written M @ v."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def dot(u, v):
    """Hand-written u . v."""
    total = 0.0
    for i in range(len(u)):
        total += u[i] * v[i]
    return total


def power_method(A, x0, tol=1e-10, max_iter=500, label="", verbose=True):
    """
    Power iteration, normalizing by the LARGEST-MAGNITUDE entry.

    y[argmax|y|] preserves the sign -- np.max(y) would return the wrong value
    for a matrix whose dominant eigenvalue is negative.
    """
    x = list(x0)
    lam_old = 0.0
    if verbose:
        print(f"  {'iter':>4} | {'lambda estimate':>18} | x (normalized)")

    for iteration in range(1, max_iter + 1):
        y = matvec(A, x)

        idx = 0                               # hand-written argmax|.|
        for j in range(1, len(y)):
            if abs(y[j]) > abs(y[idx]):
                idx = j
        lam_new = y[idx]

        x = [value / lam_new for value in y]

        if verbose and (iteration <= 3 or iteration % 10 == 0):
            print(f"  {iteration:4d} | {lam_new:18.10f} | "
                  f"{[round(value, 6) for value in x]}")

        if abs(lam_new - lam_old) < tol:
            if verbose:
                print(f"  converged after {iteration} iterations "
                      f"(tol {tol}){label}")
            return lam_new, x, iteration
        lam_old = lam_new

    return lam_old, x, max_iter


def deflate(A, lam, v_unit, verbose=True):
    """
    Hotelling deflation:  A2 = A - lambda * v_hat v_hat^T.

    v_hat MUST be unit length -- the derivation uses v_hat . v_hat = 1. Passing
    a largest-entry-normalized vector here (the power method's native output)
    is the standard way to get a silently wrong A2.
    """
    n = len(A)
    A2 = [[A[i][j] - lam * v_unit[i] * v_unit[j] for j in range(n)]
          for i in range(n)]
    if verbose:
        print(f"  A2 = A - {lam:.6f} * v1 v1^T:")
        for row in A2:
            print("    ", [round(value, 6) for value in row])
    return A2


if __name__ == "__main__":
    A = [[12, 2, 1, 0],
         [2, 5, 0, 1],
         [1, 0, 3, 1],
         [0, 1, 1, 2]]
    n = 4
    # Asymmetric starting guess: [1,1,1,1] can be orthogonal to an intermediate
    # eigenvector after deflation, which stalls the second run.
    x0 = [1.0, 2.0, 3.0, -1.0]

    print(f"  A is symmetric: {np.allclose(np.array(A), np.array(A).T)}"
          "   <- the precondition for everything below")

    print("\n" + "=" * 70)
    print("(a) POWER METHOD ON A -> DOMINANT EIGENPAIR")
    print("=" * 70)
    lam1, v1_raw, iters1 = power_method(A, x0, label=" [on A]")

    norm1 = l2_norm(v1_raw)
    v1 = [value / norm1 for value in v1_raw]
    print(f"\n  lambda1 = {lam1:.10f}")
    print(f"  v1 (largest entry = 1) = {[round(value, 6) for value in v1_raw]}")
    print(f"  manual ||v1|| = {norm1:.10f}   numpy = {np.linalg.norm(v1_raw):.10f}")
    print(f"  v1 (UNIT length)       = {[round(value, 6) for value in v1]}")
    print("    ^ deflation needs the UNIT vector, not the largest-entry one")

    print("\n" + "=" * 70)
    print("(b) DEFLATE:  A2 = A - lambda1 * v1 v1^T")
    print("=" * 70)
    A2 = deflate(A, lam1, v1)

    print("\n" + "=" * 70)
    print("(c) POWER METHOD ON A2 -> SECOND EIGENPAIR OF THE ORIGINAL A")
    print("=" * 70)
    lam2, v2_raw, iters2 = power_method(A2, x0, label=" [on A2]")
    v2 = [value / l2_norm(v2_raw) for value in v2_raw]
    print(f"\n  lambda2 = {lam2:.10f}")
    print(f"  v2 (UNIT length) = {[round(value, 6) for value in v2]}")

    print("\n" + "=" * 70)
    print("(d) VERIFICATION")
    print("=" * 70)
    A_np = np.array(A, dtype=float)
    A2_np = np.array(A2, dtype=float)

    # check 1: A2 @ v1 == 0 -- lambda1 muted. Hand-computed, then cross-checked.
    muted = matvec(A2, v1)
    manual_muted = l2_norm(muted)
    numpy_muted = float(np.linalg.norm(A2_np @ np.array(v1)))
    print(f"  A2 @ v1 = {[round(value, 10) for value in muted]}")
    print(f"    manual ||A2 v1|| = {manual_muted:.3e}   numpy = {numpy_muted:.3e}"
          "   (should be ~0)")

    # check 2: orthogonality of v1 and v2
    orth = dot(v1, v2)
    print(f"  v1 . v2 = {orth:.3e}   numpy = {float(np.array(v1) @ np.array(v2)):.3e}"
          "   (should be ~0)")

    # check 3: (lam2, v2) is an eigenpair of the ORIGINAL A, not just of A2
    Av2 = matvec(A, v2)
    r = [Av2[i] - lam2 * v2[i] for i in range(n)]
    manual_res = l2_norm(r)
    print(f"  ||A v2 - lambda2 v2|| = {manual_res:.3e}   "
          f"numpy = {float(np.linalg.norm(A_np @ np.array(v2) - lam2 * np.array(v2))):.3e}")
    print("    ^ confirms the pair belongs to A itself, not merely to A2")

    # check 4: against numpy's full eigendecomposition
    eigvals = np.linalg.eigvals(A_np).real
    order = np.argsort(-np.abs(eigvals))
    print(f"\n  numpy eigenvalues of A (descending |lambda|): "
          f"{np.round(eigvals[order], 8)}")
    print(f"  ours: lambda1 = {lam1:.8f} (gap {abs(lam1 - eigvals[order][0]):.2e}), "
          f"lambda2 = {lam2:.8f} (gap {abs(lam2 - eigvals[order][1]):.2e})")

    # free global check: the eigenvalues must sum to the trace
    trace = sum(A[i][i] for i in range(n))
    print(f"  trace(A) = {trace}  vs  sum of all numpy eigenvalues = "
          f"{np.sum(eigvals):.8f}")

    assert manual_muted < 1e-6 and abs(orth) < 1e-6 and manual_res < 1e-6
    assert abs(lam1 - eigvals[order][0]) < 1e-6
    assert abs(lam2 - eigvals[order][1]) < 1e-6
    print("\n  ALL CHECKS PASSED")


# ----------------------------------------------------------------------
# WRITTEN QUESTION -- answer
#
# "This trick relies on A being symmetric. Why -- what specifically would
#  break if A weren't?"
#
# The load-bearing fact is ORTHOGONALITY of eigenvectors belonging to distinct
# eigenvalues:  v1_hat . v_j = 0 for j != 1.
#
# For a SYMMETRIC matrix the spectral theorem guarantees this. It is not a
# lucky property of this particular A -- it is structural. That guarantee is
# what makes the second line of the derivation collapse:
#
#     A2 v_j = A v_j - lambda1 v1_hat (v1_hat . v_j)
#            = lambda_j v_j - lambda1 v1_hat * 0
#            = lambda_j v_j
#
# so every OTHER eigenpair passes through the deflation completely untouched,
# and only the v1 direction is zeroed out. A2's spectrum is exactly A's with
# lambda1 replaced by 0, which is precisely what makes "power method on A2"
# return the second-largest eigenvalue OF A.
#
# For a NON-SYMMETRIC matrix eigenvectors are not guaranteed orthogonal at all.
# If v1_hat . v2 != 0, the subtracted rank-one term also removes a chunk of the
# v1 component from A v2, so
#
#     A2 v2 = lambda_2 v2 - lambda1 (v1_hat . v2) v1_hat  !=  lambda_2 v2
#
# and v2 is no longer even an eigenvector of A2. The power method on A2 will
# still converge to *something* -- it always does -- but that something is the
# dominant eigenvalue of a matrix whose spectrum is no longer related to A's in
# any useful way.
#
# THAT is the dangerous part: there is no error, no warning, no failure to
# converge. You get a plausible-looking number that is simply not an
# eigenvalue of A. The defence is check 3 above -- always verify the exposed
# pair against the ORIGINAL matrix (||A v2 - lambda2 v2||), never against the
# deflated one. That check passes here and would fail loudly on a
# non-symmetric A.
#
# (Secondary point worth a sentence if asked: even on a symmetric matrix,
# repeated deflation accumulates error. A2 is built from the COMPUTED lambda1
# and v1, so their round-off is baked into A2, and A3 inherits it compounded.
# By the third or fourth eigenvalue the accuracy has visibly degraded -- which
# is why deflation is a two-or-three-eigenvalue tool, not a general
# eigensolver.)
# ----------------------------------------------------------------------
