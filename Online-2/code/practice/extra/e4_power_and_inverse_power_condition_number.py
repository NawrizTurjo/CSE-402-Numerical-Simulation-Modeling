"""
E4 -- Largest AND Smallest Eigenvalue -> the Condition Number    [~30 min]
==========================================================================
Topics: power method, inverse power method, condition number.
Closest real questions: C1 (power) and A1 (inverse power) -- BOTH, in one problem.
Difficulty: core.

QUESTION
--------
Given the symmetric matrix
    A = [[ 9,  2,  1,  0],
         [ 2,  7,  1,  1],
         [ 1,  1,  4,  1],
         [ 0,  1,  1,  2]]

(a) Find the DOMINANT eigenvalue and eigenvector by hand-written power
    iteration. Print every iteration's estimate.
(b) Find the SMALLEST-magnitude eigenvalue and eigenvector by hand-written
    inverse power iteration (np.linalg.inv() is allowed for A^-1 once; the
    loop must be yours). Print every iteration's estimate.
(c) A is symmetric, so its spectral condition number is
        kappa_2(A) = |lambda_max| / |lambda_min|
    Compute it from YOUR two eigenvalues, and cross-check against
    np.linalg.cond(A, 2).
(d) Verify both eigenpairs against np.linalg.eig, comparing at unit length and
    accounting for a possible sign flip.

WRITTEN QUESTION: what does the condition number tell you about solving
A x = b, and why does that make it relevant to the round-off question (P2)?

WHY THIS IS WORTH DRILLING
--------------------------
The two real eigenvalue questions faced by A1 and C1 are the same loop with one
line changed (multiply by A vs. solve with A). Doing both back to back in one
sitting is the cheapest way to make that connection stick -- and the condition
number is the natural third part that ties the eigenvalue half of the course
back to the linear-systems half.
"""
import numpy as np


def l2_norm(vec):
    """Hand-written ||v||_2."""
    total = 0.0
    for value in vec:
        total += value * value
    return total ** 0.5


def matvec(M, v):
    """Hand-written M @ v."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def argmax_abs(v):
    """Hand-written argmax|.| -- NOT np.max, which loses the sign."""
    idx = 0
    for j in range(1, len(v)):
        if abs(v[j]) > abs(v[idx]):
            idx = j
    return idx


def power_iteration(A, x0, tol=1e-10, max_iter=200, verbose=True):
    """
    Dominant eigenpair: repeatedly apply A, normalize by the largest entry.

    Converges because  A^k x0 = sum_i c_i lambda_i^k v_i, and dividing through
    by lambda_1^k leaves the v_1 term plus factors (lambda_i/lambda_1)^k -> 0.
    Convergence speed is governed by |lambda_2 / lambda_1|.
    """
    x = list(x0)
    lam_old = 0.0
    if verbose:
        print(f"  {'iter':>4} | {'lambda estimate':>18} | x (largest entry = 1)")
    for iteration in range(1, max_iter + 1):
        y = matvec(A, x)                     # y = A x
        idx = argmax_abs(y)
        lam_new = y[idx]                     # the divisor IS the estimate
        x = [value / lam_new for value in y]

        if verbose and (iteration <= 5 or iteration % 10 == 0):
            print(f"  {iteration:4d} | {lam_new:18.10f} | "
                  f"{[round(value, 6) for value in x]}")

        if abs(lam_new - lam_old) < tol:
            if verbose:
                print(f"  converged after {iteration} iterations")
            return lam_new, x, iteration
        lam_old = lam_new
    return lam_old, x, max_iter


def inverse_power_iteration(A_inv, x0, tol=1e-10, max_iter=200, verbose=True):
    """
    Smallest-magnitude eigenpair of A, via the power method on A^-1.

    If A v = lambda v then A^-1 v = (1/lambda) v -- inverting FLIPS the
    magnitude ranking, so A's smallest eigenvalue is A^-1's dominant one.
    Run the ordinary power method on A^-1, then reciprocate.

    THE BUG TO AVOID: forgetting the final 1/mu. The iteration converges just
    as nicely either way, and the number it prints looks perfectly plausible.
    """
    x = list(x0)
    lam_old = 0.0
    if verbose:
        print(f"  {'iter':>4} | {'mu = eig(A^-1)':>18} | {'1/mu = eig(A)':>16}")
    for iteration in range(1, max_iter + 1):
        y = matvec(A_inv, x)                 # y = A^-1 x
        idx = argmax_abs(y)
        mu = y[idx]                          # dominant eigenvalue OF A^-1
        x = [value / mu for value in y]
        lam_of_A = 1.0 / mu                  # <-- the reciprocal step

        if verbose and (iteration <= 5 or iteration % 10 == 0):
            print(f"  {iteration:4d} | {mu:18.10f} | {lam_of_A:16.10f}")

        if abs(lam_of_A - lam_old) < tol:
            if verbose:
                print(f"  converged after {iteration} iterations")
            return lam_of_A, x, iteration
        lam_old = lam_of_A
    return lam_old, x, max_iter


def check_eigenpair(A, lam, v_raw, name):
    """||A v - lambda v|| hand-computed, then compared against np.linalg.eig."""
    n = len(v_raw)
    A_np = np.array(A, dtype=float)

    norm = l2_norm(v_raw)
    v = [value / norm for value in v_raw]          # unit length, numpy's convention

    Av = matvec(A, v)
    r = [Av[i] - lam * v[i] for i in range(n)]
    manual = l2_norm(r)
    numpy_res = float(np.linalg.norm(A_np @ np.array(v) - lam * np.array(v)))

    eigvals, eigvecs = np.linalg.eig(A_np)
    idx = int(np.argmin(np.abs(eigvals - lam)))
    v_np = eigvecs[:, idx].real
    same = float(np.linalg.norm(np.array(v) - v_np))
    flipped = float(np.linalg.norm(np.array(v) + v_np))

    print(f"  {name}:")
    print(f"    lambda ours = {lam:.10f}   numpy = {eigvals[idx].real:.10f}   "
          f"gap = {abs(lam - eigvals[idx].real):.2e}")
    print(f"    manual ||Av - lambda v|| = {manual:.3e}   numpy = {numpy_res:.3e}")
    print(f"    v ours  (unit) = {[round(value, 6) for value in v]}")
    print(f"    v numpy (unit) = {np.round(v_np, 6)}")
    print(f"    same-sign diff = {same:.2e}, flipped = {flipped:.2e}  -> "
          f"{'SIGN FLIPPED (expected, v and -v are both valid)' if flipped < same else 'same sign'}")
    return manual, min(same, flipped), abs(lam - eigvals[idx].real)


if __name__ == "__main__":
    A = [[9, 2, 1, 0],
         [2, 7, 1, 1],
         [1, 1, 4, 1],
         [0, 1, 1, 2]]
    n = 4
    x0 = [1.0, 2.0, 3.0, -1.0]        # generic, not symmetric

    print(f"  A symmetric: {np.allclose(np.array(A), np.array(A).T)}"
          "   (so kappa_2 = |lambda_max| / |lambda_min| applies)")

    print("\n" + "=" * 70)
    print("(a) POWER ITERATION -> LARGEST |eigenvalue|")
    print("=" * 70)
    lam_max, v_max, iters_max = power_iteration(A, x0)
    print(f"\n  lambda_max = {lam_max:.10f}")

    print("\n" + "=" * 70)
    print("(b) INVERSE POWER ITERATION -> SMALLEST |eigenvalue|")
    print("=" * 70)
    # The one library call the A1-style question explicitly allows.
    A_inv = np.linalg.inv(np.array(A, dtype=float)).tolist()
    print("  A^-1 (via np.linalg.inv, as the question permits):")
    for row in A_inv:
        print("   ", [round(value, 6) for value in row])
    print()
    lam_min, v_min, iters_min = inverse_power_iteration(A_inv, x0)
    print(f"\n  lambda_min = {lam_min:.10f}")

    print("\n" + "=" * 70)
    print("(c) CONDITION NUMBER  kappa_2(A) = |lambda_max| / |lambda_min|")
    print("=" * 70)
    kappa = abs(lam_max) / abs(lam_min)
    kappa_numpy = float(np.linalg.cond(np.array(A, dtype=float), 2))
    print(f"  from OUR eigenvalues: {abs(lam_max):.10f} / {abs(lam_min):.10f} "
          f"= {kappa:.10f}")
    print(f"  np.linalg.cond(A, 2) = {kappa_numpy:.10f}")
    print(f"  relative gap = {abs(kappa - kappa_numpy) / kappa_numpy:.3e}")
    print(f"\n  kappa ~ {kappa:.2f} -- small, so A is WELL conditioned: solving")
    print(f"  A x = b here loses at most about log10({kappa:.2f}) ~ "
          f"{np.log10(kappa):.2f} decimal digits of accuracy.")

    print("\n" + "=" * 70)
    print("(d) VERIFICATION AGAINST np.linalg.eig")
    print("=" * 70)
    res_max, gap_max, val_max = check_eigenpair(A, lam_max, v_max, "dominant eigenpair")
    print()
    res_min, gap_min, val_min = check_eigenpair(A, lam_min, v_min, "smallest eigenpair")

    all_eigs = np.linalg.eigvals(np.array(A, dtype=float)).real
    print(f"\n  all eigenvalues (numpy): {np.round(np.sort(all_eigs)[::-1], 8)}")
    trace = sum(A[i][i] for i in range(n))
    print(f"  free check: trace(A) = {trace}  vs  sum of eigenvalues = "
          f"{np.sum(all_eigs):.8f}")
    print(f"  iterations: power = {iters_max}, inverse power = {iters_min}")

    assert res_max < 1e-8 and res_min < 1e-8
    assert gap_max < 1e-6 and gap_min < 1e-6
    assert val_max < 1e-6 and val_min < 1e-6
    assert abs(kappa - kappa_numpy) / kappa_numpy < 1e-8
    print("\n  ALL CHECKS PASSED")


# ----------------------------------------------------------------------
# WRITTEN QUESTION -- answer
#
# "What does the condition number tell you about solving A x = b, and why does
#  that make it relevant to the round-off question?"
#
# kappa(A) is the AMPLIFICATION FACTOR from input error to output error. The
# bound is
#
#     ||delta x|| / ||x||   <=   kappa(A) * ||delta b|| / ||b||
#
# so a relative perturbation of size eps in the data can produce a relative
# error of up to kappa * eps in the answer. A useful rule of thumb: you lose
# roughly log10(kappa) decimal digits of accuracy. With double precision giving
# about 16 digits, kappa ~ 10^3 leaves ~13 good digits, while kappa ~ 10^14
# leaves ~2 -- and kappa ~ 10^16 means the computed x can be numerically
# meaningless even though nothing crashed.
#
# For a SYMMETRIC matrix kappa_2 = |lambda_max| / |lambda_min|, which is why
# the two iterative methods above are enough to compute it: the power method
# supplies the numerator, the inverse power method the denominator. For a
# general matrix the same formula holds with SINGULAR values in place of
# eigenvalues, which is what np.linalg.cond(A, 2) actually computes -- the two
# agree here because A is symmetric.
#
# THE LINK TO THE ROUND-OFF QUESTION (P2), which is the point of asking:
# these are two DIFFERENT failure modes, and confusing them is the trap.
#
#   - P2's failure is ALGORITHMIC. A was perfectly well conditioned; naive
#     elimination chose a tiny pivot, produced a multiplier of ~917, and
#     amplified round-off by that factor. The fix is a better algorithm --
#     partial pivoting -- and it recovers the exact answer.
#
#   - Ill-conditioning is INHERENT to the matrix. No pivoting strategy, no
#     algorithm, no amount of care helps: the problem itself is sensitive, and
#     any tiny perturbation of b moves x a long way.
#
# The practical consequence for exam verification: a SMALL residual ||Ax - b||
# does NOT prove x is accurate. For an ill-conditioned A you can have a
# residual of 1e-15 and an x that is wrong in the second decimal place, because
# a wildly wrong x can still nearly satisfy the equations. Whenever a question
# hands you a suspicious-looking matrix, print np.linalg.cond(A) alongside the
# residual -- the two numbers together are what actually justify the answer.
# See code/complex_cases/ill_conditioned_systems.py for a worked example of
# exactly that failure.
# ----------------------------------------------------------------------
