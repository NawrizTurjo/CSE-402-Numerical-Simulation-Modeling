"""
Verification Helpers -- "my implementation" vs "the library", as functions
==========================================================================

EVERY past question ends the same way: *verify your result against NumPy and
report a residual / error norm.* (B1: "verify against np.linalg.solve, report
||Ax-b||_2". C1/A1: "compare with np.linalg.eig". C2: "verify ||A-LU||",
"verify with np.linalg.solve()", "calculate the residual for both".)

So it is worth having that step as a single call instead of retyping five
print statements under time pressure. One function per result type:

  | you computed              | call                                    |
  |---------------------------|-----------------------------------------|
  | a solution vector x       | verify_solution(A, b, x)                |
  | an LU factorization       | verify_lu(A, L, U)  /  ..., P=P         |
  | an inverse                | verify_inverse(A, A_inv)                |
  | a determinant             | verify_determinant(A, det_value)        |
  | one eigenpair             | verify_eigenpair(A, lam, v)             |
  | several eigenpairs        | verify_eigenpairs(A, lams, vecs)        |
  | A = V Lambda V^-1         | verify_eigendecomposition(A, V, Lambda) |
  | A^k                       | verify_matrix_power(A, k, A_k)          |

THE ORDER OF OPERATIONS MATTERS FOR MARKS (see manual_ops.py and
res/A2_Prep.md section 1): the error/residual is part of YOUR answer, so each
function below computes it **by hand first** (plain Python loops, no NumPy),
prints that as the answer, and only then prints the NumPy one-liner beside it
as an independent cross-check. Both numbers are shown; they must agree.

Every function prints a readable block and RETURNS True/False so it can also
be used inside an `assert`.

This file is deliberately self-contained -- it imports nothing from the rest of
the repo, so a single copy-paste into the exam file is enough.
"""
import numpy as np

TOL = 1e-8


# ----------------------------------------------------------------------
# Hand-written primitives (duplicated from manual_ops.py on purpose, so this
# file can be copy-pasted alone into an exam answer)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 1. LINEAR SYSTEM SOLUTION
# ----------------------------------------------------------------------

def verify_solution(A, b, x, label="Ax = b", tol=TOL, verbose=True):
    A, b, x = np.asarray(A, float), np.asarray(b, float), np.asarray(x, float)
    
    # Residual: ||Ax - b||
    res = float(np.linalg.norm(A @ x - b))
    
    # Gap vs numpy solver
    x_lib = np.linalg.solve(A, b)
    gap = float(np.linalg.norm(x - x_lib))
    
    ok = res < tol and gap < tol
    if verbose:
        print(f"\n--- Verify: {label} ---")
        print("Our x:            ", np.round(x, 6))
        print("np.linalg.solve:  ", np.round(x_lib, 6))
        print("Residual ||Ax-b||:", round(res, 8))
        print("Status:           ", "PASS" if ok else "FAIL")
    return ok


# ----------------------------------------------------------------------
# 2. LU FACTORIZATION
# ----------------------------------------------------------------------

def verify_lu(A, L, U, P=None, tol=TOL, verbose=True):
    A, L, U = np.asarray(A, float), np.asarray(L, float), np.asarray(U, float)
    target = A if P is None else np.asarray(P, float) @ A
    name = "A - LU" if P is None else "PA - LU"
    
    # Reconstruction error ||target - L@U||
    err = float(np.linalg.norm(target - L @ U))
    
    # Structure checks: L is unit lower triangular, U is upper triangular
    n = L.shape[0]
    unit_lower = np.allclose(L, np.tril(L), atol=tol) and np.allclose(np.diag(L), 1.0, atol=tol)
    upper = np.allclose(U, np.triu(U), atol=tol)
    
    ok = err < tol and unit_lower and upper
    if verbose:
        print(f"\n--- Verify: LU Factorization ({name}) ---")
        print("Reconstruction Error ||PA - LU||:", round(err, 8))
        print("L is unit lower triangular:     ", unit_lower)
        print("U is upper triangular:          ", upper)
        print("Status:                          ", "PASS" if ok else "FAIL")
    return ok


# ----------------------------------------------------------------------
# 3. MATRIX INVERSE
# ----------------------------------------------------------------------

def verify_inverse(A, A_inv, tol=TOL, verbose=True):
    A, A_inv = np.asarray(A, float), np.asarray(A_inv, float)
    n = len(A)
    
    # Error ||A @ A_inv - I||
    I_err = float(np.linalg.norm(A @ A_inv - np.eye(n)))
    
    # Gap vs numpy inverse
    lib_inv = np.linalg.inv(A)
    gap = float(np.linalg.norm(A_inv - lib_inv))
    
    ok = I_err < tol and gap < tol
    if verbose:
        print("\n--- Verify: Matrix Inverse ---")
        print("Error ||A A^-1 - I||:", round(I_err, 8))
        print("Gap vs np.linalg.inv:", round(gap, 8))
        print("Status:              ", "PASS" if ok else "FAIL")
    return ok


# ----------------------------------------------------------------------
# 4. DETERMINANT
# ----------------------------------------------------------------------

def verify_determinant(A, det_value, tol=1e-6, verbose=True):
    A = np.asarray(A, float)
    lib_det = float(np.linalg.det(A))
    err = abs(det_value - lib_det)
    scale = max(1.0, abs(lib_det))
    ok = (err / scale) < tol
    
    if verbose:
        print("\n--- Verify: Determinant ---")
        print("Our det:       ", round(det_value, 6))
        print("np.linalg.det: ", round(lib_det, 6))
        print("Status:        ", "PASS" if ok else "FAIL")
    return ok


# ----------------------------------------------------------------------
# 5. EIGENPAIRS
# ----------------------------------------------------------------------

def verify_eigenpair(A, lam, v, label=None, tol=1e-6, verbose=True):
    A = np.asarray(A, float)
    v = np.asarray(v, float)
    
    # Unit normalize vector
    v_unit = v / np.linalg.norm(v)
    
    # Residual ||A v - lam v||
    res = float(np.linalg.norm(A @ v_unit - lam * v_unit))
    
    # Compare against np.linalg.eig
    vals, vecs = np.linalg.eig(A)
    vals = vals.real
    idx = int(np.argmin(np.abs(vals - lam)))
    lam_lib = float(vals[idx])
    v_lib = vecs[:, idx].real
    
    # Gap in eigenvector (handles sign flips v vs -v)
    vec_gap = float(min(np.linalg.norm(v_unit - v_lib), np.linalg.norm(v_unit + v_lib)))
    
    ok = res < tol and abs(lam - lam_lib) < tol and vec_gap < tol
    if verbose:
        print(f"\n--- Verify: {label or 'Eigenpair'} ---")
        print("Our lambda:          ", round(lam, 6), f"(NumPy: {round(lam_lib, 6)})")
        print("Residual ||Av-lam*v||:", round(res, 8))
        print("Eigenvector Gap:     ", round(vec_gap, 8))
        print("Status:              ", "PASS" if ok else "FAIL")
    return ok


def verify_eigenpairs(A, lams, vecs, tol=1e-6, verbose=True):
    A = np.asarray(A, float)
    vecs = np.asarray(vecs, float)
    cols = [vecs[:, j] for j in range(vecs.shape[1])] if vecs.ndim == 2 else list(vecs)
    
    all_ok = True
    for i, (lam, v) in enumerate(zip(lams, cols)):
        all_ok &= verify_eigenpair(A, lam, v, label=f"Eigenpair {i+1}", tol=tol, verbose=verbose)
    
    # Trace check sum(lambda) == tr(A)
    trace_ok = abs(sum(lams) - np.trace(A)) < tol
    if verbose:
        print(f"Trace Check (sum(lambda) == tr(A)): {'PASS' if trace_ok else 'FAIL'}")
    return bool(all_ok and trace_ok)


# ----------------------------------------------------------------------
# 6. FULL EIGENDECOMPOSITION AND MATRIX POWERS
# ----------------------------------------------------------------------

def verify_eigendecomposition(A, V, Lam, tol=1e-6, verbose=True):
    A, V, Lam = np.asarray(A, float), np.asarray(V, float), np.asarray(Lam, float)
    
    # Reconstructed matrix A = V @ Lam @ V^-1
    V_inv = np.linalg.inv(V)
    A_rec = V @ Lam @ V_inv
    err = float(np.linalg.norm(A - A_rec))
    
    ok = err < tol
    if verbose:
        print("\n--- Verify: Eigendecomposition ---")
        print("Reconstruction Error ||A - V Lam V^-1||:", round(err, 8))
        print("Status:                                ", "PASS" if ok else "FAIL")
    return ok


def verify_matrix_power(A, k, A_k, tol=1e-6, verbose=True):
    A, A_k = np.asarray(A, float), np.asarray(A_k, float)
    lib_k = np.linalg.matrix_power(A, k)
    
    err = float(np.linalg.norm(A_k - lib_k))
    scale = max(1.0, float(np.linalg.norm(lib_k)))
    ok = (err / scale) < tol
    
    if verbose:
        print(f"\n--- Verify: Matrix Power A^{k} ---")
        print("Our A^k:\n", np.round(A_k, 6))
        print("np.linalg.matrix_power:\n", np.round(lib_k, 6))
        print("Status:                ", "PASS" if ok else "FAIL")
    return ok


# ----------------------------------------------------------------------
# 7. SYSTEM CLASSIFICATION (unique / no solution / infinite)
# ----------------------------------------------------------------------

def verify_classification(A, b, claimed, tol=1e-9, verbose=True):
    A, b = np.asarray(A, float), np.asarray(b, float)
    n = A.shape[1]
    r_A = int(np.linalg.matrix_rank(A))
    r_aug = int(np.linalg.matrix_rank(np.hstack([A, b.reshape(-1, 1)])))
    
    if r_A < r_aug:
        expected = "no_solution"
    elif r_A == n:
        expected = "unique"
    else:
        expected = "infinite"
    
    ok = (claimed == expected)
    if verbose:
        print("\n--- Verify: System Classification ---")
        print("Claimed:", claimed, "| Expected (Rank Test):", expected)
        print("Status: ", "PASS" if ok else "FAIL")
    return ok


# ----------------------------------------------------------------------
# SELF-TEST
# ----------------------------------------------------------------------

def _self_test():
    """Each verifier must PASS on a correct input and FAIL on a corrupted one."""
    A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]], dtype=float)
    b = np.array([7, 10, 10], dtype=float)
    x = np.linalg.solve(A, b)

    assert verify_solution(A, b, x, verbose=False)
    assert not verify_solution(A, b, x + 0.1, verbose=False)

    L = np.array([[1, 0, 0], [0.5, 1, 0], [0.25, 0.625, 1]], dtype=float)
    U = np.array([[4, 2, 1], [0, 4, 2.5], [0, 0, 4.1875]], dtype=float)
    assert verify_lu(A, L, U, verbose=False)
    assert not verify_lu(A, L, U + 0.1, verbose=False)

    assert verify_inverse(A, np.linalg.inv(A), verbose=False)
    assert not verify_inverse(A, np.linalg.inv(A) + 0.1, verbose=False)

    assert verify_determinant(A, float(np.linalg.det(A)), verbose=False)
    assert not verify_determinant(A, float(np.linalg.det(A)) + 1.0, verbose=False)

    vals, vecs = np.linalg.eig(A)
    assert verify_eigenpair(A, float(vals[0]), vecs[:, 0], verbose=False)
    assert not verify_eigenpair(A, float(vals[0]) + 0.5, vecs[:, 0], verbose=False)
    assert verify_eigenpairs(A, [float(value) for value in vals], vecs, verbose=False)

    Lam = np.diag(vals.real)
    assert verify_eigendecomposition(A, vecs.real, Lam, verbose=False)
    A5 = vecs.real @ np.diag(vals.real ** 5) @ np.linalg.inv(vecs.real)
    assert verify_matrix_power(A, 5, A5, verbose=False)

    assert verify_classification(A, b, "unique", verbose=False)
    singular = np.array([[1, 1, 1], [2, 2, 2], [1, 2, 3]], dtype=float)
    assert verify_classification(singular, np.array([1, 3, 4], dtype=float),
                                 "no_solution", verbose=False)
    assert verify_classification(singular, np.array([1, 2, 4], dtype=float),
                                 "infinite", verbose=False)

    print("verification.py: all self-tests passed "
          "(every verifier accepts correct input and rejects corrupted input)")


if __name__ == "__main__":
    _self_test()
