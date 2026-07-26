"""
Manual Linear-Algebra Primitives -- the pieces NumPy is NOT allowed to compute
==============================================================================

WHY THIS FILE EXISTS (read this, it is an exam-rule thing, not a style thing):
-----------------------------------------------------------------------------
`res/A2_Prep.md` section 1 records the rubric rule confirmed by a classmate who
saw the real marking scheme:

    Anything that is part of YOUR OWN answer must be hand-coded. NumPy is only
    allowed afterwards, to re-derive the SAME number as an independent check.

That includes several calls that look too trivial to matter, and are the easiest
marks to lose:

  | you wrote                       | why it is NOT "verification"                |
  |---------------------------------|---------------------------------------------|
  | np.linalg.norm(x)  to normalize | the norm is part of your eigenvector answer |
  | A @ x              for residual | Ax is part of your residual answer          |
  | np.linalg.norm(A - L@U)         | ||A-LU|| is the answer to "verify the LU"   |
  | np.outer(v, v)     in deflation | the deflated matrix is your answer          |

Every function below is the hand-written version of one of those. The rule of
thumb in the exam:

    manual_value = <function from this file>
    numpy_value  = <the np.linalg one-liner>
    print(f"manual = {manual_value}   numpy = {numpy_value}")   # same number

Report BOTH. The manual number is the answer; the NumPy number is the check.

THEORETICAL FORMULAS IMPLEMENTED HERE:
--------------------------------------
  l2_norm(v)          ||v||_2 = sqrt( sum_i v_i^2 )
  inf_norm(v)         ||v||_inf = max_i |v_i|
  one_norm(v)         ||v||_1 = sum_i |v_i|
  frobenius_norm(M)   ||M||_F = sqrt( sum_i sum_j M_ij^2 )   (entrywise L2)
  dot(u, v)           u . v = sum_i u_i v_i
  matvec(M, v)        (Mv)_i = sum_j M_ij v_j
  matmul(X, Y)        (XY)_ij = sum_k X_ik Y_kj
  outer(u, v)         (u v^T)_ij = u_i v_j
  transpose(M)        (M^T)_ij = M_ji
  identity(n)         I_ij = 1 if i == j else 0
  normalize(v)        v / ||v||_2          (unit length)
  normalize_max(v)    v / v[argmax |v_i|]  (largest entry becomes exactly 1)
  residual(A, x, b)   r = Ax - b           (vector, NOT its norm)
  trace(M)            tr(M) = sum_i M_ii

All of them accept either a NumPy array or a plain list-of-lists / list, and
return plain Python floats / lists -- so they work identically whether the exam
question hands you `np.array(...)` or raw `input()`-parsed lists.
"""
import numpy as np


# ----------------------------------------------------------------------
# VECTOR NORMS
# ----------------------------------------------------------------------

def l2_norm(v):
    """
    Euclidean norm  ||v||_2 = sqrt( sum_i v_i^2 ).

    NumPy equivalent (verification only): np.linalg.norm(v)
    """
    total = 0.0
    for value in v:
        total += value * value
    return total ** 0.5


def inf_norm(v):
    """
    Maximum norm  ||v||_inf = max_i |v_i|.

    NumPy equivalent (verification only): np.linalg.norm(v, np.inf)
    """
    largest = 0.0
    for value in v:
        if abs(value) > largest:
            largest = abs(value)
    return largest


def one_norm(v):
    """
    Manhattan norm  ||v||_1 = sum_i |v_i|.

    NumPy equivalent (verification only): np.linalg.norm(v, 1)
    """
    total = 0.0
    for value in v:
        total += abs(value)
    return total


# ----------------------------------------------------------------------
# MATRIX NORMS
# ----------------------------------------------------------------------

def frobenius_norm(M):
    """
    Frobenius (entrywise L2) norm  ||M||_F = sqrt( sum_i sum_j M_ij^2 ).

    This is the norm meant by "verify ||A - LU||" in the C2 question: flatten
    the difference matrix and take the ordinary L2 norm of all its entries.

    NumPy equivalent (verification only): np.linalg.norm(M)  or  ...(M, 'fro')
    """
    total = 0.0
    for row in M:
        for value in row:
            total += value * value
    return total ** 0.5


def max_abs_entry(M):
    """
    max_{i,j} |M_ij| -- the quickest "is this matrix ~zero?" check.

    NumPy equivalent (verification only): np.max(np.abs(M))
    """
    largest = 0.0
    for row in M:
        for value in row:
            if abs(value) > largest:
                largest = abs(value)
    return largest


# ----------------------------------------------------------------------
# PRODUCTS
# ----------------------------------------------------------------------

def dot(u, v):
    """
    Inner product  u . v = sum_i u_i v_i.

    NumPy equivalent (verification only): u @ v  or  np.dot(u, v)
    """
    total = 0.0
    for i in range(len(u)):
        total += u[i] * v[i]
    return total


def matvec(M, v):
    """
    Matrix-vector product  (Mv)_i = sum_j M_ij v_j.

    Row i of the result is the dot product of row i of M with v.

    NumPy equivalent (verification only): M @ v
    """
    result = []
    for row in M:
        total = 0.0
        for j in range(len(v)):
            total += row[j] * v[j]
        result.append(total)
    return result


def matmul(X, Y):
    """
    Matrix-matrix product  (XY)_ij = sum_k X_ik Y_kj.

    Triple loop: i over rows of X, j over columns of Y, k over the shared
    inner dimension.

    NumPy equivalent (verification only): X @ Y
    """
    rows = len(X)
    inner = len(Y)
    cols = len(Y[0])
    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            total = 0.0
            for k in range(inner):
                total += X[i][k] * Y[k][j]
            result[i][j] = total
    return result


def outer(u, v):
    """
    Outer product  (u v^T)_ij = u_i v_j  -- an (len(u) x len(v)) matrix.

    This is the piece deflation needs:  A2 = A - lambda * outer(v_hat, v_hat).

    NumPy equivalent (verification only): np.outer(u, v)
    """
    return [[u[i] * v[j] for j in range(len(v))] for i in range(len(u))]


# ----------------------------------------------------------------------
# MATRIX SHAPES / STRUCTURE
# ----------------------------------------------------------------------

def transpose(M):
    """
    (M^T)_ij = M_ji.

    NumPy equivalent (verification only): M.T
    """
    rows = len(M)
    cols = len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]


def identity(n):
    """
    n x n identity matrix.

    NumPy equivalent (verification only): np.eye(n)
    """
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def trace(M):
    """
    tr(M) = sum_i M_ii -- also equals the sum of all eigenvalues, which makes
    it a fast sanity check on any eigenvalue list you produce.

    NumPy equivalent (verification only): np.trace(M)
    """
    total = 0.0
    for i in range(len(M)):
        total += M[i][i]
    return total


def subtract_matrices(X, Y):
    """
    Entrywise X - Y. Needed before taking ||A - LU|| by hand.

    NumPy equivalent (verification only): X - Y
    """
    return [[X[i][j] - Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]


def subtract_vectors(u, v):
    """
    Entrywise u - v. Needed before taking ||Ax - b|| by hand.

    NumPy equivalent (verification only): u - v
    """
    return [u[i] - v[i] for i in range(len(u))]


# ----------------------------------------------------------------------
# NORMALIZATION (the two conventions that must never be mixed up)
# ----------------------------------------------------------------------

def normalize(v):
    """
    Scale v to UNIT EUCLIDEAN LENGTH:  v / ||v||_2.

    This is NumPy's eigenvector convention -- always convert your power-method
    output to this before comparing element-by-element with np.linalg.eig.

    NumPy equivalent (verification only): v / np.linalg.norm(v)
    """
    norm = l2_norm(v)
    return [value / norm for value in v]


def normalize_max(v):
    """
    Scale v so its LARGEST-MAGNITUDE ENTRY becomes exactly 1 (sign preserved):
        v / v[argmax_i |v_i|]

    This is the power-method convention -- the divisor is itself the eigenvalue
    estimate, which is why the method normalizes this way instead of by ||v||_2.

    Returns (scaled_vector, divisor).
    """
    idx = 0
    for j in range(1, len(v)):
        if abs(v[j]) > abs(v[idx]):
            idx = j
    divisor = v[idx]
    return [value / divisor for value in v], divisor


# ----------------------------------------------------------------------
# RESIDUALS (the single most-asked "verify" quantity in past questions)
# ----------------------------------------------------------------------

def residual(A, x, b):
    """
    Residual VECTOR  r = Ax - b  (not its norm -- take l2_norm(r) for that).

    Past questions ask for "the residual |Ax-b|" -- print the vector AND its
    norm; the vector shows which equation is off, the norm is the single number.

    NumPy equivalent (verification only): A @ x - b
    """
    Ax = matvec(A, x)
    return subtract_vectors(Ax, b)


def residual_norm(A, x, b):
    """
    ||Ax - b||_2 as a single number, entirely hand-computed.

    NumPy equivalent (verification only): np.linalg.norm(A @ x - b)
    """
    return l2_norm(residual(A, x, b))


def reconstruction_error(A, L, U):
    """
    ||A - LU||_F as a single number, entirely hand-computed.
    This is the exact answer C2's step 2 ("verify ||A - LU||") asks for.

    For a PIVOTED factorization PA = LU, pass matmul(P, A) as the first
    argument instead of A.

    NumPy equivalent (verification only): np.linalg.norm(A - L @ U)
    """
    return frobenius_norm(subtract_matrices(A, matmul(L, U)))


# ----------------------------------------------------------------------
# SELF-TEST -- every manual function cross-checked against its NumPy twin
# ----------------------------------------------------------------------

def _self_test():
    """Each hand-written primitive must reproduce its np.linalg counterpart."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    B = rng.normal(size=(4, 4))
    u = rng.normal(size=4)
    v = rng.normal(size=4)

    assert abs(l2_norm(v) - np.linalg.norm(v)) < 1e-12
    assert abs(inf_norm(v) - np.linalg.norm(v, np.inf)) < 1e-12
    assert abs(one_norm(v) - np.linalg.norm(v, 1)) < 1e-12
    assert abs(frobenius_norm(A) - np.linalg.norm(A, 'fro')) < 1e-12
    assert abs(max_abs_entry(A) - np.max(np.abs(A))) < 1e-12
    assert abs(dot(u, v) - u @ v) < 1e-12
    assert np.allclose(matvec(A, v), A @ v)
    assert np.allclose(matmul(A, B), A @ B)
    assert np.allclose(outer(u, v), np.outer(u, v))
    assert np.allclose(transpose(A), A.T)
    assert np.allclose(identity(4), np.eye(4))
    assert abs(trace(A) - np.trace(A)) < 1e-12
    assert np.allclose(subtract_matrices(A, B), A - B)
    assert np.allclose(subtract_vectors(u, v), u - v)
    assert np.allclose(normalize(v), v / np.linalg.norm(v))

    scaled, divisor = normalize_max(v)
    assert abs(divisor - v[np.argmax(np.abs(v))]) < 1e-12
    assert abs(max(abs(value) for value in scaled) - 1.0) < 1e-12

    # residual / reconstruction on a system with a known exact answer
    A2 = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]], dtype=float)
    b2 = np.array([7, 10, 10], dtype=float)
    x2 = np.linalg.solve(A2, b2)
    assert np.allclose(residual(A2, x2, b2), A2 @ x2 - b2, atol=1e-12)
    assert abs(residual_norm(A2, x2, b2) - np.linalg.norm(A2 @ x2 - b2)) < 1e-12

    L = np.array([[1, 0, 0], [0.5, 1, 0], [0.25, 0.625, 1]], dtype=float)
    U = np.array([[4, 2, 1], [0, 4, 2.5], [0, 0, 4.1875]], dtype=float)
    assert abs(reconstruction_error(A2, L, U)
               - np.linalg.norm(A2 - L @ U, 'fro')) < 1e-12

    print("manual_ops.py: all self-tests passed "
          "(every manual primitive matches its NumPy twin)")


if __name__ == "__main__":
    _self_test()
