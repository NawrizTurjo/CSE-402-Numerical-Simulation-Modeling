"""
E6 -- A = V Lambda V^-1: Reconstruct, Power, and Solve With It   [~30 min]
==========================================================================
Topics: eigenvalue decomposition, matrix powers, solving a system in the eigenbasis.
Closest real question: none yet (eigendecomposition is in the syllabus, untested).
Difficulty: medium.

QUESTION
--------
Given the symmetric matrix
    A = [[ 4,  1,  1],
         [ 1,  3,  0],
         [ 1,  0,  2]]
and  b = [6, 4, 3]

(a) Get all eigenpairs of A (np.linalg.eig is allowed here -- there is no
    short hand-written general eigensolver, and it is not what is being
    tested). Build V with eigenvectors as COLUMNS and Lambda = diag(lambdas),
    with column j of V paired to Lambda[j][j].
(b) Verify A = V Lambda V^-1 by hand-written matrix multiplication, reporting
    ||A - V Lambda V^-1||_F.
(c) Compute A^6 two ways -- V Lambda^6 V^-1 and np.linalg.matrix_power -- and
    confirm they agree.
(d) Solve A x = b THROUGH the decomposition:
        A x = b  ->  V Lambda V^-1 x = b  ->  x = V Lambda^-1 V^-1 b
    (Lambda^-1 is just the reciprocals of the diagonal.) Compare against
    np.linalg.solve and report ||Ax - b||_2.
(e) Also compute A^(-1) as V Lambda^-1 V^-1 and A^(1/2) as V Lambda^(1/2) V^-1,
    confirming (A^(1/2))^2 == A.

WRITTEN QUESTION: solving A x = b via the eigendecomposition works, but nobody
does it in practice. Why not -- and when IS the decomposition the right tool?

WHY THIS IS WORTH DRILLING
--------------------------
Eigendecomposition is the shortest topic on the syllabus and the most likely to
be asked as a "build V and Lambda, reconstruct A, compute A^k" question. Part
(d) is the piece that makes it click: once you are in the eigenbasis, ANY
function of A -- inverse, power, square root -- is that function applied to n
scalars on the diagonal.
"""
import numpy as np


def matmul(X, Y):
    """Hand-written X @ Y."""
    rows, inner, cols = len(X), len(Y), len(Y[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            total = 0.0
            for k in range(inner):
                total += X[i][k] * Y[k][j]
            out[i][j] = total
    return out


def matvec(M, v):
    """Hand-written M @ v."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def frobenius_norm(M):
    """Hand-written ||M||_F."""
    total = 0.0
    for row in M:
        for value in row:
            total += value * value
    return total ** 0.5


def l2_norm(vec):
    """Hand-written ||v||_2."""
    total = 0.0
    for value in vec:
        total += value * value
    return total ** 0.5


def diagonal_function(eigenvalues, f):
    """
    Build f(Lambda) -- a diagonal matrix with f applied to each eigenvalue.

    This one helper covers EVERY function of A:
        f = lambda t: t ** k      ->  A^k
        f = lambda t: 1 / t       ->  A^-1
        f = lambda t: t ** 0.5    ->  the matrix square root
    Because Lambda is diagonal, "apply f to the matrix" degenerates to "apply f
    to n scalars" -- no matrix multiplication in the f step at all.
    """
    n = len(eigenvalues)
    return [[f(eigenvalues[i]) if i == j else 0.0 for j in range(n)]
            for i in range(n)]


if __name__ == "__main__":
    A = [[4, 1, 1],
         [1, 3, 0],
         [1, 0, 2]]
    b = [6, 4, 3]
    n = 3

    A_np = np.array(A, dtype=float)
    print(f"  A symmetric: {np.allclose(A_np, A_np.T)}   "
          "(so A is guaranteed diagonalizable -- spectral theorem)")

    print("\n" + "=" * 70)
    print("(a) BUILD V AND Lambda")
    print("=" * 70)
    evals, evecs = np.linalg.eig(A_np)
    eigenvalues = [float(value) for value in evals.real]

    # np.linalg.eig already returns eigenvectors as COLUMNS -- evecs[:, j] pairs
    # with evals[j]. Keep that pairing; do NOT transpose or re-sort one alone.
    V = [[float(evecs[i, j].real) for j in range(n)] for i in range(n)]
    Lam = diagonal_function(eigenvalues, lambda t: t)
    V_inv = np.linalg.inv(np.array(V)).tolist()

    for j in range(n):
        column = [V[i][j] for i in range(n)]
        print(f"  lambda{j + 1} = {eigenvalues[j]:12.8f}   v{j + 1} (column {j}) = "
              f"{[round(value, 6) for value in column]}")
    print(f"\n  V (columns are eigenvectors) =")
    for row in V:
        print("   ", [round(value, 6) for value in row])
    print(f"  Lambda =")
    for row in Lam:
        print("   ", [round(value, 6) for value in row])

    print("\n" + "=" * 70)
    print("(b) VERIFY A = V Lambda V^-1")
    print("=" * 70)
    reconstructed = matmul(matmul(V, Lam), V_inv)
    print("  V Lambda V^-1 =")
    for row in reconstructed:
        print("   ", [round(value, 10) for value in row])
    diff = [[A[i][j] - reconstructed[i][j] for j in range(n)] for i in range(n)]
    manual_err = frobenius_norm(diff)
    numpy_err = float(np.linalg.norm(
        A_np - np.array(V) @ np.array(Lam) @ np.array(V_inv), 'fro'))
    print(f"  manual ||A - V Lam V^-1||_F = {manual_err:.3e}   numpy = {numpy_err:.3e}")

    print("\n" + "=" * 70)
    print("(c) A^6 VIA V Lambda^6 V^-1")
    print("=" * 70)
    k = 6
    Lam_k = diagonal_function(eigenvalues, lambda t: t ** k)
    print(f"  Lambda^{k} diagonal = "
          f"{[round(Lam_k[i][i], 4) for i in range(n)]}   "
          f"({n} scalar powers -- no matrix multiplication)")
    A_k = matmul(matmul(V, Lam_k), V_inv)
    A_k_numpy = np.linalg.matrix_power(A_np, k)
    print(f"  A^{k} (ours)  =\n{np.round(np.array(A_k), 6)}")
    print(f"  A^{k} (numpy) =\n{np.round(A_k_numpy, 6)}")
    rel_err = frobenius_norm([[A_k[i][j] - A_k_numpy[i][j] for j in range(n)]
                              for i in range(n)]) / frobenius_norm(A_k_numpy.tolist())
    print(f"  relative error = {rel_err:.3e}")

    print("\n" + "=" * 70)
    print("(d) SOLVE A x = b THROUGH THE DECOMPOSITION")
    print("=" * 70)
    print("  A x = b  ->  V Lam V^-1 x = b  ->  x = V Lam^-1 V^-1 b")
    print("  three cheap steps, no elimination anywhere:")
    c = matvec(V_inv, b)                      # 1. b into eigen-coordinates
    print(f"    1. c = V^-1 b            = {[round(value, 6) for value in c]}"
          "   (b expressed in the eigenbasis)")
    d = [c[i] / eigenvalues[i] for i in range(n)]   # 2. divide by each eigenvalue
    print(f"    2. d = Lam^-1 c          = {[round(value, 6) for value in d]}"
          "   (just n scalar divisions)")
    x = matvec(V, d)                          # 3. back to standard coordinates
    print(f"    3. x = V d               = {[round(value, 8) for value in x]}"
          "   (back to standard coordinates)")

    Ax = matvec(A, x)
    residual = [Ax[i] - b[i] for i in range(n)]
    manual_res = l2_norm(residual)
    numpy_res = float(np.linalg.norm(A_np @ np.array(x) - np.array(b, dtype=float)))
    x_lib = np.linalg.solve(A_np, np.array(b, dtype=float))
    print(f"\n  np.linalg.solve(A, b) = {np.round(x_lib, 8)}")
    print(f"  manual ||Ax-b||_2 = {manual_res:.3e}   numpy = {numpy_res:.3e}")

    print("\n" + "=" * 70)
    print("(e) OTHER FUNCTIONS OF A, FOR FREE")
    print("=" * 70)
    A_inv = matmul(matmul(V, diagonal_function(eigenvalues, lambda t: 1.0 / t)), V_inv)
    print("  A^-1 = V Lam^-1 V^-1  (reciprocate the eigenvalues):")
    for row in A_inv:
        print("   ", [round(value, 8) for value in row])
    print(f"  np.linalg.inv(A) =\n{np.round(np.linalg.inv(A_np), 8)}")

    A_sqrt = matmul(matmul(V, diagonal_function(eigenvalues, lambda t: t ** 0.5)), V_inv)
    print("\n  A^(1/2) = V Lam^(1/2) V^-1  (square-root the eigenvalues):")
    for row in A_sqrt:
        print("   ", [round(value, 8) for value in row])
    squared = matmul(A_sqrt, A_sqrt)
    sqrt_err = frobenius_norm([[squared[i][j] - A[i][j] for j in range(n)]
                               for i in range(n)])
    print(f"  (A^(1/2))^2 =")
    for row in squared:
        print("   ", [round(value, 8) for value in row])
    print(f"  ||(A^(1/2))^2 - A||_F = {sqrt_err:.3e}")
    print("\n  (all eigenvalues are positive here, so the real square root exists;")
    print("   a matrix with a negative eigenvalue would give a complex A^(1/2))")

    assert manual_err < 1e-10
    assert rel_err < 1e-10
    assert np.allclose(x, x_lib, atol=1e-9) and manual_res < 1e-9
    assert np.allclose(A_inv, np.linalg.inv(A_np), atol=1e-9)
    assert sqrt_err < 1e-10
    print("\n  ALL CHECKS PASSED")


# ----------------------------------------------------------------------
# WRITTEN QUESTION -- answer
#
# "Solving A x = b via the eigendecomposition works, but nobody does it in
#  practice. Why not -- and when IS the decomposition the right tool?"
#
# WHY NOT, FOR SOLVING:
#
# 1. COST. Computing the eigendecomposition is far more expensive than LU.
#    Eigenvalues have no closed form past degree 4 (Abel-Ruffini), so every
#    real eigensolver is ITERATIVE -- np.linalg.eig runs the QR algorithm,
#    typically ~10x the flops of an LU factorization, with a runtime that
#    depends on the matrix rather than only on n. LU is one direct, predictable
#    (2/3)n^3 sweep. Paying 10x to solve one system is a bad trade.
#
# 2. IT NEEDS MORE THAN LU DOES. LU (with pivoting) works for ANY non-singular
#    matrix. The eigendecomposition needs A to be DIAGONALIZABLE -- n
#    independent eigenvectors, or V is singular and V^-1 does not exist. A
#    defective matrix like [[2,1],[0,2]] has no eigendecomposition at all, yet
#    LU solves systems with it happily. And a merely NEAR-defective A gives a
#    nearly-singular V, so V^-1 is huge and the answer is swamped by round-off.
#
# 3. IT CAN BE COMPLEX FOR NO REASON. A real non-symmetric A can have complex
#    eigenvalues, forcing complex arithmetic through the whole solve even
#    though the answer x is entirely real.
#
# WHEN IT *IS* THE RIGHT TOOL -- whenever you need the same matrix applied
# MANY times, or applied as a FUNCTION:
#
#   - A^k for large or repeated k: V Lam^k V^-1 costs the same for k = 6 and
#     k = 10^6 (part (c) above). Markov chains and discrete dynamical systems
#     live here.
#   - Any f(A): A^-1, A^(1/2), exp(A) -- part (e). Once diagonalized, f(A) is
#     just f applied to n scalars. There is no LU analogue of this at all.
#   - Understanding BEHAVIOUR rather than getting a number: the eigenvalues
#     say whether a system is stable (all |lambda| < 1), which mode dominates
#     as k grows, how ill-conditioned A is (kappa = |lam_max|/|lam_min| for
#     symmetric A -- see E4), and how fast the power method will converge
#     (|lam_2/lam_1|).
#
# The rule of thumb: LU is for ANSWERS, eigendecomposition is for STRUCTURE
# and for repeated application. Solving a single A x = b through eigenvalues is
# a valid derivation and a bad algorithm.
# ----------------------------------------------------------------------
