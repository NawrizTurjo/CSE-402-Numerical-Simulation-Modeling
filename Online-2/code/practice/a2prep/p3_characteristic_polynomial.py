"""
P3 -- Direct Eigenvalues via the Characteristic Polynomial [~25 min]  res/A2_Prep.md sec. 10
=============================================================================================

QUESTION
--------
Given  A = [[6, 2], [2, 3]]:

(a) Derive the characteristic polynomial det(A - lambda I) = 0 in terms of A's
    trace and determinant (lambda^2 - trace*lambda + det = 0 for a 2x2).
(b) Solve for both eigenvalues with the quadratic formula.
(c) For each eigenvalue, hand-solve (A - lambda I) v = 0 for its eigenvector
    (2x2, so one row gives the ratio v1:v2 directly) and normalize to unit
    length.
(d) Verify against np.linalg.eig(A).

WHAT THIS DRILLS
----------------
The opposite style from A1/C1: no iteration at all, straight algebra. Worth
having in your fingers because it is the only method that gives ALL eigenvalues
of a small matrix exactly, in one shot, with no convergence to wait for -- and
because the discriminant tells you in advance whether the power method would
even converge on this matrix.

THE DERIVATION, ONCE:
    A v = lambda v  and  v != 0
    (A - lambda I) v = 0  has a nonzero solution
    <=> (A - lambda I) is singular
    <=> det(A - lambda I) = 0
For A = [[a,b],[c,d]]:
    det(A - lambda I) = (a-lambda)(d-lambda) - bc
                      = lambda^2 - (a+d) lambda + (ad - bc)
                      = lambda^2 - trace(A) lambda + det(A)
"""
import math
import numpy as np


def characteristic_polynomial_2x2(A, verbose=True):
    """
    Returns (trace, det, discriminant) for lambda^2 - trace*lambda + det = 0.
    """
    a, b = A[0][0], A[0][1]
    c, d = A[1][0], A[1][1]

    trace = a + d                      # sum of the diagonal
    det = a * d - b * c                # ad - bc
    disc = trace * trace - 4 * det     # the quadratic's b^2 - 4ac

    if verbose:
        print(f"  A = [[{a}, {b}], [{c}, {d}]]")
        print(f"  trace(A) = {a} + {d} = {trace}")
        print(f"  det(A)   = {a}*{d} - {b}*{c} = {det}")
        print(f"\n  det(A - lambda I) = ({a} - L)({d} - L) - ({b})({c})")
        print(f"                    = lambda^2 - {trace}*lambda + {det} = 0")
        print(f"\n  discriminant = {trace}^2 - 4*{det} = {disc}")
        if disc > 0:
            print("    > 0  -> two distinct REAL eigenvalues")
        elif disc == 0:
            print("    = 0  -> one REPEATED real eigenvalue")
        else:
            print("    < 0  -> a COMPLEX-CONJUGATE pair "
                  "(the power method would oscillate forever on this matrix)")
    return trace, det, disc


def eigenvalues_quadratic_formula(trace, det, disc, verbose=True):
    """lambda = [ trace +/- sqrt(disc) ] / 2."""
    root = math.sqrt(disc)
    lam1 = (trace + root) / 2
    lam2 = (trace - root) / 2
    if verbose:
        print(f"  lambda = [{trace} +/- sqrt({disc})] / 2 = "
              f"[{trace} +/- {root}] / 2")
        print(f"  lambda1 = {lam1}")
        print(f"  lambda2 = {lam2}")
    return lam1, lam2


def eigenvector_for(A, lam, verbose=True):
    """
    Solves (A - lambda I) v = 0 by hand for a 2x2.

    B = A - lambda I is SINGULAR by construction, so its two rows are
    proportional -- one row carries all the information. Row 1 reads
        B00 * v1 + B01 * v2 = 0   ->   v2 / v1 = -B00 / B01
    so v = (1, -B00/B01) up to scale. If B01 == 0 that row says nothing about
    the ratio, so fall back to row 2 the same way.
    """
    b00 = A[0][0] - lam
    b01 = A[0][1]
    b10 = A[1][0]
    b11 = A[1][1] - lam

    if abs(b01) > 1e-12:
        v = [1.0, -b00 / b01]
        used = f"row 1:  ({b00:.6g})v1 + ({b01:.6g})v2 = 0  ->  v2/v1 = {-b00 / b01:.6g}"
    elif abs(b10) > 1e-12:
        v = [-b11 / b10, 1.0]
        used = f"row 2:  ({b10:.6g})v1 + ({b11:.6g})v2 = 0  ->  v1/v2 = {-b11 / b10:.6g}"
    else:
        v = [1.0, 0.0] if abs(b00) < 1e-12 else [0.0, 1.0]
        used = "A - lambda I is diagonal -- eigenvector is a coordinate axis"

    # manual L2 norm -- normalizing our own vector is part of OUR answer
    norm = math.sqrt(v[0] * v[0] + v[1] * v[1])
    v_unit = [v[0] / norm, v[1] / norm]

    if verbose:
        print(f"  lambda = {lam}:  A - lambda I = "
              f"[[{b00:.6g}, {b01:.6g}], [{b10:.6g}, {b11:.6g}]]")
        print(f"    {used}")
        print(f"    v (unnormalized) = {[round(value, 6) for value in v]}")
        print(f"    manual ||v|| = {norm:.10f}   numpy = {np.linalg.norm(v):.10f}")
        print(f"    v (unit)         = {[round(value, 6) for value in v_unit]}")
    return v_unit


if __name__ == "__main__":
    A = [[6, 2],
         [2, 3]]

    print("=" * 70)
    print("(a) CHARACTERISTIC POLYNOMIAL")
    print("=" * 70)
    trace, det, disc = characteristic_polynomial_2x2(A)

    print("\n" + "=" * 70)
    print("(b) EIGENVALUES VIA THE QUADRATIC FORMULA")
    print("=" * 70)
    lam1, lam2 = eigenvalues_quadratic_formula(trace, det, disc)

    print("\n  free sanity checks (cost nothing, catch everything):")
    print(f"    sum  = {lam1 + lam2}  vs  trace(A) = {trace}")
    print(f"    prod = {lam1 * lam2}  vs  det(A)   = {det}")

    print("\n" + "=" * 70)
    print("(c) EIGENVECTORS FROM (A - lambda I) v = 0")
    print("=" * 70)
    v1 = eigenvector_for(A, lam1)
    print()
    v2 = eigenvector_for(A, lam2)

    print("\n" + "=" * 70)
    print("(d) VERIFICATION")
    print("=" * 70)
    A_np = np.array(A, dtype=float)

    # The definitional check: ||A v - lambda v||, hand-computed then cross-checked
    for lam, v, name in [(lam1, v1, "lambda1"), (lam2, v2, "lambda2")]:
        Av = [A[0][0] * v[0] + A[0][1] * v[1],
              A[1][0] * v[0] + A[1][1] * v[1]]
        r = [Av[0] - lam * v[0], Av[1] - lam * v[1]]
        manual = math.sqrt(r[0] * r[0] + r[1] * r[1])
        numpy_res = float(np.linalg.norm(A_np @ np.array(v) - lam * np.array(v)))
        print(f"  {name} = {lam}:  A@v = {[round(value, 6) for value in Av]}, "
              f"lambda*v = {[round(lam * value, 6) for value in v]}")
        print(f"    manual ||Av - lambda v|| = {manual:.3e}   numpy = {numpy_res:.3e}")

    eigvals, eigvecs = np.linalg.eig(A_np)
    print(f"\n  numpy eigenvalues  = {eigvals}")
    print(f"  numpy eigenvectors (columns) =\n{np.round(eigvecs, 8)}")

    for lam, v, name in [(lam1, v1, "v1"), (lam2, v2, "v2")]:
        idx = int(np.argmin(np.abs(eigvals - lam)))
        v_np = eigvecs[:, idx]
        same = float(np.linalg.norm(np.array(v) - v_np))
        flipped = float(np.linalg.norm(np.array(v) + v_np))
        print(f"  {name}: same-sign diff = {same:.2e}, flipped = {flipped:.2e}  -> "
              f"{'SIGN FLIPPED vs numpy (expected, v and -v are both valid)' if flipped < same else 'same sign'}")

    assert np.allclose(sorted([lam1, lam2]), sorted(eigvals.real))
    print("\n  ALL CHECKS PASSED")


# ----------------------------------------------------------------------
# NOTES WORTH REMEMBERING
#
# - For a 2x2, memorize this and you never need to expand a determinant:
#       lambda^2 - trace(A)*lambda + det(A) = 0
#   The 3x3 version generalizes to
#       lambda^3 - tr(A) lambda^2 + (sum of 2x2 principal minors) lambda
#           - det(A) = 0
#   (see code/algorithms/characteristic_polynomial.py for the 3x3 implementation).
#
# - sum(lambda_i) == trace(A) and prod(lambda_i) == det(A) hold for ANY n.
#   Two free lines of verification on every eigenvalue question in the course,
#   including the iterative ones -- if deflation gives you four eigenvalues
#   that do not sum to the trace, one of them is wrong.
#
# - The discriminant is a convergence PREDICTOR for the iterative methods:
#   disc < 0 means complex-conjugate eigenvalues of EQUAL magnitude, so
#   |lambda_2 / lambda_1| = 1 and the power method never converges. If the
#   power method oscillates on a 2x2, computing the discriminant explains why
#   in one line.
#
# - v and -v are both valid eigenvectors, so a sign mismatch against NumPy is
#   expected and is not an error -- flagged in the original C1 question too.
# ----------------------------------------------------------------------
