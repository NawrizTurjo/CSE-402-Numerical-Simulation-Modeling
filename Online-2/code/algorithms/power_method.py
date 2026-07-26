"""
Power Method -- Canonical Library Functions & Theoretical Mappings
===================================================================

THEORETICAL ALGORITHM MAPPING:
------------------------------
Computes the dominant eigenvalue lambda_1 and associated eigenvector v_1 of matrix A (n x n).

1. DOMINANT EIGENVALUE ASSUMPTION:
   Requires |lambda_1| > |lambda_2| >= |lambda_3| >= ... >= |lambda_n|.
   (Strictly larger in absolute magnitude than all other eigenvalues).

2. POWER ITERATION DERIVATION:
   Expand initial guess vector x^(0) in terms of A's eigenvector basis:
     x^(0) = c_1 * v_1 + c_2 * v_2 + ... + c_n * v_n
   Applying matrix A k times gives:
     A^k * x^(0) = c_1 * lambda_1^k * v_1 + c_2 * lambda_2^k * v_2 + ...
                 = lambda_1^k * [ c_1 * v_1 + sum_{i=2}^n c_i * (lambda_i / lambda_1)^k * v_i ]
   Since |lambda_i / lambda_1| < 1 for all i >= 2, as k -> infinity, (lambda_i / lambda_1)^k -> 0.
   Therefore, A^k * x^(0) converges directly to a scalar multiple of dominant eigenvector v_1!

3. RECURRENCE STEP AT ITERATION k:
   a) Unnormalized matrix-vector product:
      y^(k) = A * x^(k-1)

   b) Dominant Eigenvalue Estimate (magnitude-based argmax):
      idx = argmax_{j} |y^(k)_j|
      lambda^(k) = y^(k)[idx]    (Preserves correct sign for negative eigenvalues!)

   c) Vector Normalization (L-infinity norm):
      x^(k) = y^(k) / lambda^(k)    (Ensures max element magnitude is 1.0)

   d) Convergence Check:
      max_j |x^(k)_j - x^(k-1)_j| < tol
"""
import numpy as np


def power_iteration(A, x0, tol=1e-8, max_iter=1000, hand_written=True, verbose=False):
    """
    Computes dominant eigenvalue and unit-normalized eigenvector via Power Iteration.

    THEORETICAL STEPS:
    1. Initialize vector x = x0.
    2. Loop up to max_iter:
       y = A @ x
       lambda_est = element of y with max magnitude
       x_new = y / lambda_est
       check convergence: ||x_new - x||_inf < tol
    3. Normalize final eigenvector v to unit L2-norm: v = x / ||x||_2.

    Returns:
        dominant_eigenvalue (float): Convergence estimate lambda_1.
        unit_eigenvector (ndarray): Normalized eigenvector (L2-norm = 1.0).
        history (list): Sequence of eigenvalue estimates across iterations.
    """
    x = x0.astype(float).copy()
    history = []

    for it in range(max_iter):
        # Step a: Unnormalized matrix-vector multiplication y = A * x
        y = A @ x

        # Step b: Estimate eigenvalue using entry with maximum magnitude
        # Using argmax on abs(y) handles negative dominant eigenvalues correctly
        if hand_written:
            idx = 0
            for j in range(1, len(y)):
                if abs(y[j]) > abs(y[idx]):
                    idx = j
        else:
            idx = np.argmax(np.abs(y))
        lam_est = y[idx]
        history.append(lam_est)

        # Step c: Normalize iterate vector by dominant eigenvalue estimate
        x_new = y / lam_est

        if verbose:
            print(f"  iter {it + 1:3d}: eig est = {lam_est:12.6f}")

        # Step d: Convergence check using L-infinity norm (max absolute difference)
        if hand_written:
            max_diff = 0.0
            for j in range(len(x)):
                diff = abs(x_new[j] - x[j])
                if diff > max_diff:
                    max_diff = diff
        else:
            max_diff = np.max(np.abs(x_new - x))

        if max_diff < tol:
            x = x_new
            break

        x = x_new

    # Final Step: Normalize eigenvector to unit L2 length (||x||_2 = 1.0)
    if hand_written:
        norm_sq = 0.0
        for val in x:
            norm_sq += val * val
        x = x / (norm_sq ** 0.5)
    else:
        x = x / np.linalg.norm(x)
    return history[-1], x, history


def is_oscillating(history, tail=8, tol=1e-4):
    """
    Heuristic test for non-convergence / oscillation symptoms in Power Iteration.

    THEORETICAL CAUSE OF OSCILLATION:
    - Dominant eigenvalues are complex conjugates (a +/- bi).
    - Equal magnitude, opposite sign dominant eigenvalues (lambda_1 = -lambda_2).
    In these cases, (lambda_2 / lambda_1)^k does NOT approach zero, causing sign flips.
    """
    if len(history) < tail + 1:
        return False
    recent = history[-tail:]
    # Count sign flips between consecutive iterations
    sign_changes = sum(1 for i in range(len(recent) - 1) if recent[i] * recent[i + 1] < 0)
    not_converged = abs(recent[-1] - recent[-2]) > tol
    return sign_changes >= tail // 2 and not_converged


def verify_power_method(A, lam, v, tol=1e-6, verbose=True):
    A, v = np.asarray(A, float), np.asarray(v, float)
    
    # 1. Unit normalize vector
    v_unit = v / np.linalg.norm(v)
    
    # 2. Residual ||Av - lam*v||
    res = float(np.linalg.norm(A @ v_unit - lam * v_unit))
    
    # 3. Dominant eigenpair from np.linalg.eig
    vals, vecs = np.linalg.eig(A)
    dom = int(np.argmax(np.abs(vals)))
    lam_lib = float(vals[dom].real)
    v_lib = vecs[:, dom].real
    
    # 4. Eigenvector gap (handles sign flips v vs -v)
    vec_gap = float(min(np.linalg.norm(v_unit - v_lib), np.linalg.norm(v_unit + v_lib)))
    
    ok = res < tol and abs(lam - lam_lib) < tol and vec_gap < tol
    if verbose:
        print("\n--- Verify: Power Method (Dominant Eigenpair) ---")
        print("Our lambda:          ", round(lam, 6), f"(np.linalg.eig: {round(lam_lib, 6)})")
        print("Residual ||Av-lam*v||:", round(res, 8))
        print("Eigenvector Gap:     ", round(vec_gap, 8))
        print("Status:              ", "PASS" if ok else "FAIL")
    return ok


def _self_test():
    """Self-test power iteration on 4x4 matrix with dominant eigenvalue 10.0."""
    A = np.array([[8, 2, 0, 0], [2, 8, 0, 0], [0, 0, 3, 1], [0, 0, 1, 3]], dtype=float)
    x0 = np.array([1, 1, 1, 1], dtype=float)
    lam, v, hist = power_iteration(A, x0)
    assert abs(lam - 10.0) < 1e-6, lam
    assert not is_oscillating(hist)
    assert verify_power_method(A, lam, v, verbose=False)

    # A sign-flipped eigenvector must still pass -- v and -v are both valid
    assert verify_power_method(A, lam, -v, verbose=False)
    # A wrong eigenvalue must not
    assert not verify_power_method(A, lam + 0.5, v, verbose=False)

    # Negative dominant eigenvalue: catches the np.max(y) vs y[argmax|y|] bug,
    # which would return +9 here instead of the true -12.
    B = np.array([[-12, 0, 0], [0, 9, 0], [0, 0, 2]], dtype=float)
    lam_b, v_b, _ = power_iteration(B, np.array([1, 1, 1], dtype=float))
    assert abs(lam_b + 12.0) < 1e-6, lam_b
    assert verify_power_method(B, lam_b, v_b, verbose=False)

    print("power_method.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
