"""
E5 -- Deflating TWICE: the Third Eigenvalue, and Where Accuracy Goes  [~35 min]
===============================================================================
Topics: power method, Hotelling deflation applied repeatedly, error accumulation.
Closest real question: C1 (power method) extended. Difficulty: hardest -- last topic taught.

QUESTION
--------
Given the symmetric matrix
    A = [[ 6,  2,  1,  0],
         [ 2,  5,  1,  1],
         [ 1,  1,  4,  1],
         [ 0,  1,  1,  3]]

(a) Find (lambda1, v1) by hand-written power iteration.
(b) Deflate:  A2 = A - lambda1 v1 v1^T.  Find (lambda2, v2) from A2.
(c) Deflate AGAIN: A3 = A2 - lambda2 v2 v2^T. Find (lambda3, v3) from A3.
(d) Recover lambda4 for free from the trace identity, without any iteration:
        lambda4 = trace(A) - lambda1 - lambda2 - lambda3
(e) Verify all four against np.linalg.eig, and report how the error in each
    successive eigenvalue grows.

WRITTEN QUESTION: accuracy degrades with each successive deflation (watch the
eigenpair residuals grow in part (e)). Explain the mechanism, and say what it
implies about using deflation to find ALL eigenvalues of a large matrix.

WHY THIS IS WORTH DRILLING
--------------------------
P5 does ONE deflation; this does two, which is where the interesting behaviour
appears -- the accumulating error, and the trace shortcut that gets you the
last eigenvalue with no iteration at all. Also drills the starting-vector trap:
a symmetric guess like [1,1,1,1] can be orthogonal to an intermediate
eigenvector, which stalls the second or third run for no visible reason.
"""
import numpy as np


def l2_norm(vec):
    total = 0.0
    for value in vec:
        total += value * value
    return total ** 0.5


def matvec(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def dot(u, v):
    total = 0.0
    for i in range(len(u)):
        total += u[i] * v[i]
    return total


def power_iteration(A, x0, tol=1e-12, max_iter=2000):
    """Dominant eigenpair; returns (lambda, UNIT eigenvector, iterations)."""
    x = list(x0)
    lam_old = 0.0
    for iteration in range(1, max_iter + 1):
        y = matvec(A, x)
        idx = 0
        for j in range(1, len(y)):
            if abs(y[j]) > abs(y[idx]):
                idx = j
        lam_new = y[idx]
        x = [value / lam_new for value in y]
        if abs(lam_new - lam_old) < tol:
            norm = l2_norm(x)
            return lam_new, [value / norm for value in x], iteration
        lam_old = lam_new
    norm = l2_norm(x)
    return lam_old, [value / norm for value in x], max_iter


def deflate(A, lam, v_unit):
    """
    A_next = A - lambda * v_hat v_hat^T,  with ||v_hat||_2 == 1.

    Unit length is REQUIRED: the derivation uses v_hat . v_hat = 1. Feeding in
    the power method's raw largest-entry-normalized vector produces a
    plausible-looking but wrong deflated matrix.
    """
    n = len(A)
    return [[A[i][j] - lam * v_unit[i] * v_unit[j] for j in range(n)]
            for i in range(n)]


if __name__ == "__main__":
    A = [[6, 2, 1, 0],
         [2, 5, 1, 1],
         [1, 1, 4, 1],
         [0, 1, 1, 3]]
    n = 4
    # NOT [1,1,1,1]: a symmetric guess can be orthogonal to an intermediate
    # eigenvector, which silently stalls the second or third power run.
    x0 = [1.0, 2.0, 3.0, -1.0]

    print(f"  A symmetric: {np.allclose(np.array(A), np.array(A).T)}"
          "   <- required, or deflation is invalid (see P5's written answer)")
    trace = sum(A[i][i] for i in range(n))
    print(f"  trace(A) = {trace}  <- the eigenvalues must sum to this")

    print("\n" + "=" * 70)
    print("(a) FIRST EIGENPAIR FROM A")
    print("=" * 70)
    lam1, v1, it1 = power_iteration(A, x0)
    print(f"  lambda1 = {lam1:.12f}   ({it1} iterations)")
    print(f"  v1 (unit) = {[round(value, 8) for value in v1]}")

    print("\n" + "=" * 70)
    print("(b) DEFLATE ONCE -> SECOND EIGENPAIR")
    print("=" * 70)
    A2 = deflate(A, lam1, v1)
    print("  A2 = A - lambda1 v1 v1^T:")
    for row in A2:
        print("   ", [round(value, 6) for value in row])
    lam2, v2, it2 = power_iteration(A2, x0)
    print(f"\n  lambda2 = {lam2:.12f}   ({it2} iterations)")
    print(f"  v2 (unit) = {[round(value, 8) for value in v2]}")
    print(f"  check ||A2 v1|| = {l2_norm(matvec(A2, v1)):.3e}  (lambda1 muted)")

    print("\n" + "=" * 70)
    print("(c) DEFLATE AGAIN -> THIRD EIGENPAIR")
    print("=" * 70)
    A3 = deflate(A2, lam2, v2)
    print("  A3 = A2 - lambda2 v2 v2^T:")
    for row in A3:
        print("   ", [round(value, 6) for value in row])
    lam3, v3, it3 = power_iteration(A3, x0)
    print(f"\n  lambda3 = {lam3:.12f}   ({it3} iterations)")
    print(f"  v3 (unit) = {[round(value, 8) for value in v3]}")
    print(f"  check ||A3 v1|| = {l2_norm(matvec(A3, v1)):.3e}")
    print(f"  check ||A3 v2|| = {l2_norm(matvec(A3, v2)):.3e}  (both muted)")

    print("\n" + "=" * 70)
    print("(d) FOURTH EIGENVALUE FOR FREE, FROM THE TRACE")
    print("=" * 70)
    lam4 = trace - lam1 - lam2 - lam3
    print(f"  sum(lambda_i) == trace(A) holds for ANY square matrix, so")
    print(f"  lambda4 = {trace} - {lam1:.8f} - {lam2:.8f} - {lam3:.8f}")
    print(f"          = {lam4:.12f}      (no iteration at all)")
    print("\n  Worth knowing: on an n x n matrix, deflation only has to run n-1")
    print("  times. The last eigenvalue is whatever the trace has left over.")

    print("\n" + "=" * 70)
    print("(e) VERIFICATION AND ERROR GROWTH")
    print("=" * 70)
    A_np = np.array(A, dtype=float)
    true_vals = np.sort(np.linalg.eigvals(A_np).real)[::-1]
    ours = [lam1, lam2, lam3, lam4]

    print(f"  numpy eigenvalues (descending): {np.round(true_vals, 10)}")
    print(f"\n  {'':<10}{'ours':>18}{'numpy':>18}{'abs error':>14}")
    errors = []
    for i, (mine, true) in enumerate(zip(ours, true_vals), start=1):
        err = abs(mine - true)
        errors.append(err)
        print(f"  lambda{i}   {mine:18.12f}{true:18.12f}{err:14.3e}")

    # residual check for the three eigenvectors, against the ORIGINAL A
    print()
    residuals = []
    for i, (lam, v) in enumerate([(lam1, v1), (lam2, v2), (lam3, v3)], start=1):
        Av = matvec(A, v)
        r = [Av[j] - lam * v[j] for j in range(n)]
        manual = l2_norm(r)
        residuals.append(manual)
        numpy_res = float(np.linalg.norm(A_np @ np.array(v) - lam * np.array(v)))
        print(f"  ||A v{i} - lambda{i} v{i}|| = {manual:.3e}   numpy = {numpy_res:.3e}"
              "   (checked against the ORIGINAL A, not the deflated one)")

    print(f"\n  pairwise orthogonality (guaranteed only because A is symmetric):")
    print(f"    v1 . v2 = {dot(v1, v2):+.3e}")
    print(f"    v1 . v3 = {dot(v1, v3):+.3e}")
    print(f"    v2 . v3 = {dot(v2, v3):+.3e}")

    print(f"\n  ACCURACY DEGRADATION, deflation by deflation:")
    print(f"    eigenvalue error:  {errors[0]:.1e} -> {errors[1]:.1e} -> "
          f"{errors[2]:.1e}  (grows roughly, not strictly -- individual")
    print(f"                       eigenvalues can get lucky)")
    print(f"    eigenpair residual: {residuals[0]:.1e} -> {residuals[1]:.1e} -> "
          f"{residuals[2]:.1e}  (the cleaner signal: monotone)")
    print(f"  Each deflation inherits the previous one's round-off and adds its")
    print(f"  own. At n = 4 the drift is still ~1e-12 and harmless; the point is")
    print(f"  the direction of travel, which is what rules deflation out as a")
    print(f"  general eigensolver for large n.")

    for i in range(4):
        assert errors[i] < 1e-6, (i, errors[i])
    assert abs(dot(v1, v2)) < 1e-6 and abs(dot(v2, v3)) < 1e-6
    print("\n  ALL CHECKS PASSED")


# ----------------------------------------------------------------------
# WRITTEN QUESTION -- answer
#
# "Accuracy degrades with each successive deflation. Explain the mechanism,
#  and say what it implies about using deflation to find ALL eigenvalues of a
#  large matrix."
#
# (Read the trend off the RESIDUALS ||A v_k - lambda_k v_k||, which grow
# monotonically above. Individual eigenvalue errors bounce around -- one can
# happen to land closer than its predecessor -- so they show the trend only on
# average.)
#
# THE MECHANISM: deflation is built from its own output, so it compounds.
#
#   A2 = A - lambda1_computed * v1_computed v1_computed^T
#
# Note what goes into that expression: not the true lambda1 and v1, but the
# ones the power method just produced, each carrying its own convergence error
# (~tol) plus floating-point round-off. So A2 is not exactly "A with lambda1
# replaced by 0" -- it is that matrix plus a small perturbation E1. Every
# eigenvalue of A2 is therefore slightly off, including lambda2, and the error
# in lambda2 is bounded by the error already baked into A2.
#
# Then A3 = A2 - lambda2_computed v2_computed v2^T inherits E1 AND adds its own
# E2. lambda3 is extracted from a matrix that is now two perturbations away
# from the truth. The errors do not cancel; they accumulate roughly additively,
# often worse when eigenvalues are close together (a small perturbation can
# move nearly-degenerate eigenvalues a long way, and it scrambles their
# eigenVECTORS much more than their eigenvalues).
#
# A second, subtler contribution: the theory needs v1 . v_j = 0 exactly. Our
# computed v1 is only orthogonal to the others to within tolerance, so the
# subtracted rank-one term does not leave the other eigenvectors perfectly
# untouched -- it nudges them, and that nudge is what E1 physically is.
#
# WHAT IT IMPLIES: deflation is a TWO-OR-THREE-EIGENVALUE TOOL, not a general
# eigensolver. It is the right answer for "find the second-largest eigenvalue",
# which is exactly how the syllabus uses it. It is the wrong answer for "find
# the spectrum of a 1000x1000 matrix", where by eigenvalue 20 the accumulated
# error would dominate the answer. Real eigensolvers (the QR algorithm, which
# is what np.linalg.eig calls) work on the whole matrix at once with
# orthogonal similarity transforms that do not degrade this way.
#
# THREE PRACTICAL DEFENCES, all used above:
#   1. Always verify the exposed pair against the ORIGINAL A -- compute
#      ||A v_k - lambda_k v_k||, never ||A_k v_k - lambda_k v_k||. The deflated
#      matrix will happily confirm its own drift.
#   2. Use the trace identity to get the LAST eigenvalue instead of deflating
#      one more time. It is exact arithmetic on numbers you already have, so it
#      is more accurate than another deflation would be -- though note it also
#      absorbs the accumulated error of the earlier ones, so it is a shortcut,
#      not a free lunch.
#   3. Tighten the tolerance on the EARLY power runs. Error in lambda1
#      propagates into everything downstream, so it is the cheapest place to
#      buy accuracy.
# ----------------------------------------------------------------------
