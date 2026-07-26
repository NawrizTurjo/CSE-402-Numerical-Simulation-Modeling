"""
Characteristic Polynomial -- Direct (Non-Iterative) Eigenvalues
================================================================

The odd one out among the eigenvalue methods here: power method, inverse power
method and deflation are all ITERATIVE (guess, multiply, normalize, repeat).
This file is the ALGEBRAIC route -- solve det(A - lambda*I) = 0 exactly. It is
the definition the lecture starts from, and the natural thing to be asked for
on a small matrix where iteration would be overkill.

THEORETICAL ALGORITHM MAPPING:
------------------------------
1. THE DEFINITION:
     A v = lambda v,  v != 0
     (A - lambda I) v = 0  has a nonzero solution v
     <=> (A - lambda I) is singular
     <=> det(A - lambda I) = 0                        <- characteristic equation

2. 2x2 CASE -- expand the determinant directly:
     A = [[a, b], [c, d]]
     det(A - lambda I) = (a - lambda)(d - lambda) - b*c
                       = lambda^2 - (a + d) lambda + (ad - bc)
                       = lambda^2 - tr(A) lambda + det(A)
   So for ANY 2x2:
     lambda^2 - tr(A) * lambda + det(A) = 0
     lambda = [ tr +/- sqrt(tr^2 - 4 det) ] / 2       <- quadratic formula
   Discriminant sign tells you the whole story:
     > 0  two distinct real eigenvalues
     = 0  one repeated real eigenvalue
     < 0  a complex-conjugate pair (this is exactly the case where the power
          method oscillates forever instead of converging)

3. 3x3 CASE -- the cubic, via invariants:
     det(A - lambda I) = -lambda^3 + c2 lambda^2 - c1 lambda + c0
   with
     c2 = tr(A)                                    (sum of eigenvalues)
     c1 = sum of the three 2x2 principal minors    (sum of pairwise products)
     c0 = det(A)                                   (product of eigenvalues)
   Multiply through by -1 to get the monic form used below:
     lambda^3 - tr(A) lambda^2 + M lambda - det(A) = 0
   The roots are then found with np.roots (a polynomial root-finder, NOT an
   eigensolver -- the eigen-specific work is all in building the coefficients).

4. EIGENVECTOR FOR A KNOWN lambda -- solve (A - lambda I) v = 0:
   B = A - lambda I is singular by construction, so it has a nontrivial null
   vector. Row-reduce B, set the free variable to 1, back-substitute for the
   rest, then normalize.

5. SANITY CHECKS THAT COST NOTHING:
     sum(lambda_i) == tr(A)        product(lambda_i) == det(A)
   Both are free and catch a dropped or duplicated root instantly.
"""
import numpy as np

EPS = 1e-9


# ----------------------------------------------------------------------
# 2x2 -- fully closed form, no library root-finder needed
# ----------------------------------------------------------------------

def char_poly_2x2(A, verbose=False):
    """
    Coefficients of  lambda^2 - tr*lambda + det = 0  for a 2x2 matrix.

    Returns:
        trace (float), det (float), discriminant (float)
    """
    A = np.asarray(A, dtype=float)
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]

    trace = a + d                       # sum of the diagonal
    det = a * d - b * c                 # ad - bc, computed by hand
    disc = trace * trace - 4.0 * det    # b^2 - 4ac of the quadratic

    if verbose:
        print(f"  trace = {trace}, det = {det}")
        print(f"  characteristic polynomial: lambda^2 - {trace}*lambda + {det} = 0")
        print(f"  discriminant = {disc:.6g}  -> "
              f"{'two distinct real' if disc > EPS else 'one repeated real' if abs(disc) <= EPS else 'complex conjugate pair'}"
              " eigenvalue(s)")
    return trace, det, disc


def eigenvalues_2x2(A, verbose=False):
    """
    Both eigenvalues of a 2x2 via the quadratic formula.

    THEORETICAL FORMULA:
        lambda = [ tr(A) +/- sqrt(tr(A)^2 - 4 det(A)) ] / 2

    Returns a list of two values -- floats when the discriminant is >= 0,
    complex numbers when it is negative (a genuine complex-conjugate pair, not
    an error: that matrix simply has no real eigenvalues, and the power method
    would oscillate on it forever).
    """
    trace, det, disc = char_poly_2x2(A, verbose=verbose)

    if disc >= 0:
        root = disc ** 0.5
        lam1 = (trace + root) / 2.0
        lam2 = (trace - root) / 2.0
    else:
        root = complex(0.0, (-disc) ** 0.5)
        lam1 = (trace + root) / 2.0
        lam2 = (trace - root) / 2.0

    if verbose:
        print(f"  lambda1 = {lam1}, lambda2 = {lam2}")
    return [lam1, lam2]


def eigenvector_2x2(A, lam):
    """
    Unit eigenvector of a 2x2 for a KNOWN eigenvalue, by hand.

    THEORETICAL STEP:
    B = A - lambda I is singular, so its two rows are proportional -- one row
    is enough. Row 1 says  B00*v1 + B01*v2 = 0, i.e. v2/v1 = -B00/B01.
    Take v = (1, -B00/B01) and normalize. If B01 == 0 the first row carries no
    information about the ratio, so fall back to row 2 the same way.
    """
    A = np.asarray(A, dtype=float)
    b00 = A[0, 0] - lam
    b01 = A[0, 1]
    b10 = A[1, 0]
    b11 = A[1, 1] - lam

    if abs(b01) > EPS:
        v = [1.0, -b00 / b01]
    elif abs(b10) > EPS:
        v = [-b11 / b10, 1.0]
    else:
        # Both off-diagonals vanish: A - lam I is diagonal, so the eigenvector
        # is whichever axis has a zero on the diagonal.
        v = [1.0, 0.0] if abs(b00) < EPS else [0.0, 1.0]

    norm = (v[0] * v[0] + v[1] * v[1]) ** 0.5     # hand-written L2 norm
    return [v[0] / norm, v[1] / norm]


# ----------------------------------------------------------------------
# 3x3 -- coefficients by hand, roots via a polynomial solver
# ----------------------------------------------------------------------

def char_poly_3x3(A, verbose=False):
    """
    Coefficients of the monic characteristic cubic of a 3x3 matrix:

        lambda^3 - c2 lambda^2 + c1 lambda - c0 = 0

    where
        c2 = tr(A)                                  = sum lambda_i
        c1 = sum of the three 2x2 principal minors  = sum_{i<j} lambda_i lambda_j
        c0 = det(A)                                 = prod lambda_i

    Returns the coefficient list [1, -c2, c1, -c0] in np.roots order (highest
    power first), plus the invariants (c2, c1, c0) themselves.
    """
    A = np.asarray(A, dtype=float)

    # c2 = trace, by hand
    c2 = A[0, 0] + A[1, 1] + A[2, 2]

    # c1 = sum of the 2x2 principal minors (delete row+column i for each i)
    m00 = A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]
    m11 = A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0]
    m22 = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    c1 = m00 + m11 + m22

    # c0 = det(A), by cofactor expansion along the first row
    c0 = (A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
          - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
          + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]))

    if verbose:
        print(f"  invariants: tr(A) = {c2}, sum of principal minors = {c1}, det(A) = {c0}")
        print(f"  characteristic polynomial: "
              f"lambda^3 - {c2}*lambda^2 + {c1}*lambda - {c0} = 0")
    return [1.0, -c2, c1, -c0], (c2, c1, c0)


def eigenvalues_3x3(A, verbose=False):
    """
    All three eigenvalues of a 3x3 by rooting the characteristic cubic.

    np.roots is a general POLYNOMIAL root-finder, not an eigensolver -- all the
    eigenvalue-specific work (building the invariants) is hand-coded above, so
    this stays within the "hand-write the method" rule the same way A1's
    question allows np.linalg.inv() as one named building block.

    Returns eigenvalues sorted by descending magnitude (dominant first), as
    plain floats when all roots are real.
    """
    coeffs, (c2, c1, c0) = char_poly_3x3(A, verbose=verbose)
    roots = np.roots(coeffs)

    if np.max(np.abs(roots.imag)) < 1e-9:
        roots = roots.real
    order = np.argsort(-np.abs(roots))
    roots = roots[order]

    if verbose:
        print(f"  eigenvalues (descending |lambda|) = {np.round(roots, 8)}")
        print(f"  check  sum = {np.sum(roots).real:.8g} vs tr(A) = {c2:.8g}")
        print(f"  check  prod = {np.prod(roots).real:.8g} vs det(A) = {c0:.8g}")
    return [complex(r) if np.iscomplexobj(roots) else float(r) for r in roots]


def eigenvector_via_nullspace(A, lam, hand_written=True):
    """
    Unit eigenvector for a KNOWN eigenvalue of an n x n matrix, by solving
    (A - lambda I) v = 0 with hand-written row reduction.

    THEORETICAL STEPS:
    1. B = A - lambda I  (singular by construction).
    2. Row-reduce B with partial pivoting; a column with no usable pivot is a
       FREE column (there is always at least one, since B is singular).
    3. Set the first free variable to 1, back-substitute upward for the pivot
       variables, then normalize to unit length.

    Returns a list of n floats with ||v||_2 == 1.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    B = A - lam * np.eye(n)

    pivot_of_row = {}
    free_cols = []
    row = 0

    for col in range(n):
        if row < n:
            p = row
            for i in range(row + 1, n):
                if abs(B[i, col]) > abs(B[p, col]):
                    p = i
        else:
            p = None

        if p is None or abs(B[p, col]) < 1e-8:
            free_cols.append(col)
            continue

        if p != row:
            for j in range(n):
                B[row, j], B[p, j] = B[p, j], B[row, j]

        pivot = B[row, col]
        for j in range(n):
            B[row, j] /= pivot
        for i in range(n):
            if i != row and abs(B[i, col]) > 1e-12:
                factor = B[i, col]
                for j in range(n):
                    B[i, j] -= factor * B[row, j]

        pivot_of_row[row] = col
        row += 1

    if not free_cols:
        # Numerically the matrix did not come out singular -- lam is off, or
        # the eigenvalue is a repeated one that lost precision.
        raise ValueError(f"(A - {lam}I) came out nonsingular -- lambda is wrong "
                         "or badly conditioned")

    # Free variable = 1, everything else read straight off the RREF rows
    v = [0.0] * n
    v[free_cols[0]] = 1.0
    for r, pivot_col in pivot_of_row.items():
        total = 0.0
        for j in free_cols:
            total += B[r, j] * v[j]
        v[pivot_col] = -total

    norm = 0.0                                   # hand-written L2 norm
    for value in v:
        norm += value * value
    norm = norm ** 0.5
    return [value / norm for value in v]


# ----------------------------------------------------------------------
# VERIFICATION AGAINST THE LIBRARY
# ----------------------------------------------------------------------

def verify_characteristic_polynomial(A, lams, vecs=None, tol=1e-6, verbose=True):
    A = np.asarray(A, float)
    
    # 1. Trace check sum(lambda) == tr(A)
    ok_trace = abs(complex(sum(lams)).real - np.trace(A)) < tol
    # 2. Determinant check prod(lambda) == det(A)
    lib_det = float(np.linalg.det(A))
    ok_det = abs(complex(np.prod(lams)).real - lib_det) < max(tol, tol * abs(lib_det))
    
    ok = ok_trace and ok_det
    
    # 3. Eigenvalues match np.linalg.eig
    lib_vals, _ = np.linalg.eig(A)
    for lam in lams:
        idx = int(np.argmin(np.abs(lib_vals - lam)))
        gap = abs(complex(lib_vals[idx]) - complex(lam))
        ok &= (gap < tol)
        
    # 4. Eigenvectors check if supplied
    if vecs is not None:
        for lam, v in zip(lams, vecs):
            v_arr = np.asarray(v, float)
            res = float(np.linalg.norm(A @ v_arr - complex(lam).real * v_arr))
            ok &= (res < tol)
            
    if verbose:
        print("\n--- Verify: Characteristic Polynomial ---")
        print("Trace Check (sum(lambda) == tr(A)):   ", "PASS" if ok_trace else "FAIL")
        print("Det Check (prod(lambda) == det(A)):   ", "PASS" if ok_det else "FAIL")
        print("Status:                                ", "PASS" if ok else "FAIL")
    return bool(ok)


def _self_test():
    """2x2 closed form, 3x3 cubic, complex pair, and the nullspace solver."""
    # --- 2x2 with clean integer eigenvalues (P3's matrix: lambda = 7, 2) ---
    A = np.array([[6, 2], [2, 3]], dtype=float)
    lams = eigenvalues_2x2(A)
    assert np.allclose(sorted(lams, reverse=True), [7.0, 2.0]), lams
    vecs = [eigenvector_2x2(A, lam) for lam in lams]
    assert verify_characteristic_polynomial(A, lams, vecs, verbose=False)

    # --- 2x2 with a complex-conjugate pair (rotation matrix) ---
    R = np.array([[0, -1], [1, 0]], dtype=float)
    lams_c = eigenvalues_2x2(R)
    assert all(isinstance(lam, complex) for lam in lams_c), lams_c
    assert abs(lams_c[0] - 1j) < 1e-12 and abs(lams_c[1] + 1j) < 1e-12

    # --- 3x3, eigenvalues via the cubic's invariants ---
    B = np.array([[4, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=float)
    lams3 = eigenvalues_3x3(B)
    assert np.allclose(sorted(lams3), sorted(np.linalg.eigvals(B).real), atol=1e-8)
    vecs3 = [eigenvector_via_nullspace(B, lam) for lam in lams3]
    assert verify_characteristic_polynomial(B, lams3, vecs3, verbose=False)

    # --- nullspace eigenvector on a matrix with a repeated eigenvalue ---
    C = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 5]], dtype=float)
    v = eigenvector_via_nullspace(C, 5.0)
    assert np.allclose(np.abs(v), [0, 0, 1], atol=1e-9), v

    print("characteristic_polynomial.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
