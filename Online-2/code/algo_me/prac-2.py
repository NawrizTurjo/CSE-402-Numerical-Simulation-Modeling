import numpy as np

# ==============================================================================
# TOPIC 1: GAUSS-JORDAN ELIMINATION (Multi-RHS & Matrix Inversion)
# ==============================================================================

def gauss_jordan_solver_trj(A, B, hand_written=True):
    """
    Solves A X = B using Gauss-Jordan elimination (reducing [A | B] to [I | X]).
    - B can be a single vector b (n x 1), multiple RHS vectors [b1 | b2] (n x m),
      or the Identity matrix I (n x n) for matrix inversion.
    """
    A = A.astype(float).copy()
    B = B.astype(float).copy()
    n = A.shape[0]

    # Ensure B is a 2D matrix of shape (n, m)
    if B.ndim == 1:
        B = B.reshape(-1, 1)
    m = B.shape[1]

    # Form augmented matrix [A | B] of shape n x (n + m)
    M = np.hstack([A, B])

    for k in range(n):
        # 1. Partial Pivoting: find row with largest pivot in column k
        pivot = k
        if hand_written:
            for i in range(k + 1, n):
                if abs(M[i, k]) > abs(M[pivot, k]):
                    pivot = i
        else:
            pivot = k + np.argmax(np.abs(M[k:, k]))

        if pivot != k:
            if hand_written:
                for j in range(n + m):
                    M[k, j], M[pivot, j] = M[pivot, j], M[k, j]
            else:
                M[[k, pivot]] = M[[pivot, k]]

        pivot_val = M[k, k]
        if abs(pivot_val) < 1e-12:
            raise ValueError("Zero pivot encountered; matrix is singular.")

        # 2. Normalize pivot row k so M[k, k] becomes 1.0
        if hand_written:
            for j in range(n + m):
                M[k, j] /= pivot_val
        else:
            M[k] /= pivot_val

        # 3. Eliminate column k in all other rows (both above and below row k)
        for i in range(n):
            if i != k:
                factor = M[i, k]
                if hand_written:
                    for j in range(n + m):
                        M[i, j] -= factor * M[k, j]
                else:
                    M[i] -= factor * M[k]

    # Extract solution X from right side of M: [I | X]
    X = M[:, n:]
    if X.shape[1] == 1:
        X = X.flatten()
    return X, M


def matrix_inverse_gj_trj(A, hand_written=True):
    """
    Computes A^-1 by running Gauss-Jordan reduction on [A | I] -> [I | A^-1].
    Does NOT use np.linalg.inv().
    """
    n = A.shape[0]
    I = np.eye(n)
    A_inv, _ = gauss_jordan_solver_trj(A, I, hand_written=hand_written)
    return A_inv


# ==============================================================================
# TOPIC 2: SPECTRAL RECONSTRUCTION (A = V Lambda V^-1) & 2x2 CLOSED-FORM INVERSE
# ==============================================================================

def inverse_2x2_trj(V):
    """
    Calculates closed-form 2x2 matrix inverse:
    V^-1 = 1 / (ad - bc) * [[d, -b], [-c, a]]
    """
    a, b = V[0, 0], V[0, 1]
    c, d = V[1, 0], V[1, 1]
    det = a * d - b * c
    if abs(det) < 1e-12:
        raise ValueError("Matrix is singular (determinant is zero).")
    V_inv = (np.array(
        [
            [d,-b],
            [-c,a]
        ], dtype=float
    )) * (1.0/det)
        
    return V_inv


def spectral_reconstruct_trj(V, Lambda, V_inv, hand_written=True):
    """
    Reconstructs A_rec = (V @ Lambda) @ V_inv using 3-nested for loops.
    """
    n = V.shape[0]
    if hand_written:
        # Step 1: temp = V @ Lambda
        temp = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    temp[i, j] += V[i, k] * Lambda[k, j]

        # Step 2: A_rec = temp @ V_inv
        A_rec = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    A_rec[i, j] += temp[i, k] * V_inv[k, j]
        return A_rec
    else:
        return V @ Lambda @ V_inv


def frobenius_norm_diff_trj(A, A_rec, hand_written=True):
    """
    Calculates ||A - A_rec||_F = sqrt( sum_i sum_j (A_ij - A_rec_ij)^2 ) element-by-element.
    """
    n = A.shape[0]
    if hand_written:
        total = 0.0
        for i in range(n):
            for j in range(n):
                diff = A[i, j] - A_rec[i, j]
                total += diff * diff
        return total ** 0.5
    else:
        return float(np.linalg.norm(A - A_rec, 'fro'))


# ==============================================================================
# TOPIC 3: SYSTEM CLASSIFICATION LOGIC (Unique vs. Infinite vs. No Solution)
# ==============================================================================

def classify_and_solve_trj(A, b, free_var_val=5.0, hand_written=True):
    """
    Classifies linear system Ax = b into:
    1. Unique Solution
    2. Infinite Solutions (assigns free variable x_n = free_var_val)
    3. No Solution (inconsistent system)
    """
    A = A.astype(float).copy()
    b = b.astype(float).copy().flatten()
    n = len(b)

    # Build augmented matrix [A | b]
    M = np.column_stack([A, b])

    # Forward elimination with partial pivoting
    for k in range(n - 1):
        pivot = k
        for i in range(k + 1, n):
            if abs(M[i, k]) > abs(M[pivot, k]):
                pivot = i
        if pivot != k:
            if hand_written:
                for j in range(n + 1):
                    M[k, j], M[pivot, j] = M[pivot, j], M[k, j]
            else:
                M[[k, pivot]] = M[[pivot, k]]

        if abs(M[k, k]) > 1e-12:
            for i in range(k + 1, n):
                m = M[i, k] / M[k, k]
                if hand_written:
                    for j in range(k, n + 1):
                        M[i, j] -= m * M[k, j]
                else:
                    M[i, k:] -= m * M[k, k:]

    # Classify system by checking rows for zeros
    status = "Unique Solution"
    free_var_idx = -1

    for i in range(n - 1, -1, -1):
        lhs_zero = np.all(np.abs(M[i, :n]) < 1e-10)
        rhs_zero = abs(M[i, n]) < 1e-10

        if lhs_zero and not rhs_zero:
            status = "No Solution"
            return status, None
        elif lhs_zero and rhs_zero:
            status = "Infinite Solutions"
            free_var_idx = i

    if status == "Unique Solution":
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if hand_written:
                total_sum = 0.0
                for j in range(i + 1, n):
                    total_sum += M[i, j] * x[j]
                x[i] = (M[i, n] - total_sum) / M[i, i]
            else:
                x[i] = (M[i, n] - M[i, i + 1:n] @ x[i + 1:n]) / M[i, i]
        return status, x

    elif status == "Infinite Solutions":
        x = np.zeros(n)
        # Assign arbitrary chosen value to the free variable
        x[free_var_idx] = free_var_val

        for i in range(free_var_idx - 1, -1, -1):
            if hand_written:
                total_sum = 0.0
                for j in range(i + 1, n):
                    total_sum += M[i, j] * x[j]
                x[i] = (M[i, n] - total_sum) / M[i, i]
            else:
                x[i] = (M[i, n] - M[i, i + 1:n] @ x[i + 1:n]) / M[i, i]
        return status, x


# ==============================================================================
# TOPIC 4: SHIFTED INVERSE POWER METHOD (Target Scalar s)
# ==============================================================================

# Re-use your helper PLU functions from prac.py pattern
def plu_decompose_trj(A, hand_written=True):
    A = A.astype(float).copy()
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)

    for k in range(n - 1):
        pivot = k
        if hand_written:
            for i in range(k + 1, n):
                if abs(U[i, k]) > abs(U[pivot, k]):
                    pivot = i
        else:
            pivot = k + np.argmax(np.abs(U[k:, k]))

        if pivot != k:
            if hand_written:
                for j in range(n):
                    U[k, j], U[pivot, j] = U[pivot, j], U[k, j]
                    P[k, j], P[pivot, j] = P[pivot, j], P[k, j]
                if k > 0:
                    for j in range(k):
                        L[k, j], L[pivot, j] = L[pivot, j], L[k, j]
            else:
                U[[k, pivot]] = U[[pivot, k]]
                P[[k, pivot]] = P[[pivot, k]]
                if k > 0:
                    L[[k, pivot], :k] = L[[pivot, k], :k]

        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            if hand_written:
                for j in range(k, n):
                    U[i, j] -= m * U[k, j]
            else:
                U[i, k:] -= m * U[k, k:]

    return P, L, U


def for_sub(L, P, b, hand_written=True):
    b = np.asarray(b, dtype=float).flatten()
    Pb = P @ b
    n = len(Pb)
    z = np.zeros(n)
    for i in range(n):
        if hand_written:
            total_sum = 0.0
            for j in range(i):
                total_sum += L[i, j] * z[j]
            z[i] = (Pb[i] - total_sum) / L[i, i]
        else:
            z[i] = (Pb[i] - L[i, :i] @ z[:i]) / L[i, i]
    return z


def back_sub(U, z, hand_written=True):
    n = len(z)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if hand_written:
            total_sum = 0.0
            for j in range(i + 1, n):
                total_sum += U[i, j] * x[j]
            x[i] = (z[i] - total_sum) / U[i, i]
        else:
            x[i] = (z[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def power_iteration_plu_generalized(A, x0, tol=1e-9, max_iter=1000, hand_written=True, is_inverse=True):
    x = x0.astype(float).copy()
    prev_est = None
    history = []

    if is_inverse:
        P, L, U = plu_decompose_trj(A, hand_written=hand_written)

    for _ in range(max_iter):
        if is_inverse:
            z = for_sub(L, P, x, hand_written=hand_written)
            y = back_sub(U, z, hand_written=hand_written)
        else:
            y = A @ x

        if hand_written:
            curr_est = y[0]
            for i in range(1, len(y)):
                if abs(y[i]) > abs(curr_est):
                    curr_est = y[i]
        else:
            curr_est = y[np.argmax(np.abs(y))]

        history.append(curr_est)

        if abs(curr_est) < tol:
            raise ValueError("Zero eigenvalue estimate encountered.")

        if prev_est is not None:
            ea = abs((curr_est - prev_est) / curr_est) * 100
        else:
            ea = None

        prev_est = curr_est

        x_new = y / curr_est

        if hand_written:
            max_diff = 0.0
            for i in range(len(x)):
                diff = abs(x[i] - x_new[i])
                if diff > max_diff:
                    max_diff = diff
        else:
            max_diff = np.max(np.abs(x - x_new))

        x = x_new

        if max_diff < tol:
            break

    if hand_written:
        norm_sq = 0.0
        for val in x:
            norm_sq += val * val
        x = x / (norm_sq ** 0.5)
    else:
        x = x / np.linalg.norm(x)

    if is_inverse:
        curr_est = 1.0 / curr_est

    return curr_est, x, ea, history


def shifted_inverse_power_trj(A, x0, target_shift, tol=1e-9, max_iter=1000, hand_written=True):
    """
    Finds the eigenvalue of A closest to target_shift scalar 's' using Shifted Inverse Power Iteration.
    Adjustment: A_shifted = A - s * I
    Recovery:   lambda_closest = s + lambda_shifted
    """
    n = A.shape[0]
    A_shifted = A.astype(float).copy()
    for i in range(n):
        A_shifted[i, i] -= target_shift

    # Run inverse power iteration on (A - s * I)
    eval_shifted, evec, ea, history = power_iteration_plu_generalized(
        A_shifted, x0, tol=tol, max_iter=max_iter, hand_written=hand_written, is_inverse=True
    )

    # Recover the eigenvalue closest to target_shift
    closest_eval = target_shift + eval_shifted
    return closest_eval, evec, ea, history


# ==============================================================================
# MAIN TEST DRIVER
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("1. GAUSS-JORDAN ELIMINATION (Multi-RHS & Matrix Inversion)")
    print("=" * 60)

    A_gj = np.array([
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ], dtype=float)

    # Multi-RHS: b1 = [8, -11, -3] and b2 = [1, 2, 3]
    B_multi = np.array([
        [8, 1],
        [-11, 2],
        [-3, 3]
    ], dtype=float)

    X_multi, _ = gauss_jordan_solver_trj(A_gj, B_multi, hand_written=True)
    print("Multi-RHS Solutions [x1 | x2]:\n", np.round(X_multi, 6))

    # Matrix Inversion via [A | I] -> [I | A^-1]
    A_inv_gj = matrix_inverse_gj_trj(A_gj, hand_written=True)
    print("\nMatrix Inverse via Gauss-Jordan:\n", np.round(A_inv_gj, 6))
    print("Check (A @ A^-1 == I):", np.allclose(A_gj @ A_inv_gj, np.eye(3)))

    print("\n" + "=" * 60)
    print("2. SPECTRAL RECONSTRUCTION (A = V Lambda V^-1) & 2x2 INVERSE")
    print("=" * 60)

    A_2x2 = np.array([
        [2, 1],
        [1, 2]
    ], dtype=float)

    evals, V = np.linalg.eig(A_2x2)
    Lambda = np.diag(evals)

    # Closed-form 2x2 inverse of V
    V_inv_2x2 = inverse_2x2_trj(V)
    print("V =\n", np.round(V, 6))
    print("V^-1 (Closed-form 2x2):\n", np.round(V_inv_2x2, 6))

    # Triple-loop reconstruction A_rec = (V Lambda) V^-1
    A_rec = spectral_reconstruct_trj(V, Lambda, V_inv_2x2, hand_written=True)
    print("Reconstructed A:\n", np.round(A_rec, 6))

    # Frobenius norm difference
    frob_err = frobenius_norm_diff_trj(A_2x2, A_rec, hand_written=True)
    print("Frobenius Norm Error ||A - A_rec||_F:", round(frob_err, 8))

    print("\n" + "=" * 60)
    print("3. SYSTEM CLASSIFICATION LOGIC (Unique vs Infinite vs No Solution)")
    print("=" * 60)

    # Unique System
    A_uniq = np.array([[2, 1], [1, 3]], dtype=float)
    b_uniq = np.array([5, 5], dtype=float)
    status1, x1 = classify_and_solve_trj(A_uniq, b_uniq, hand_written=True)
    print(f"System 1 Classification: {status1} | x = {np.round(x1, 6)}")

    # Infinite Solutions System (Row 2 is 0 = 0)
    A_inf = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1]], dtype=float)
    b_inf = np.array([6, 12, 3], dtype=float)
    status2, x2 = classify_and_solve_trj(A_inf, b_inf, free_var_val=5.0, hand_written=True)
    print(f"System 2 Classification: {status2} | x (x3=5) = {np.round(x2, 6)}")

    # No Solution System (Row 2 is 0 = 5)
    A_none = np.array([[1, 1], [2, 2]], dtype=float)
    b_none = np.array([2, 5], dtype=float)
    status3, x3 = classify_and_solve_trj(A_none, b_none, hand_written=True)
    print(f"System 3 Classification: {status3} | x = {x3}")

    print("\n" + "=" * 60)
    print("4. SHIFTED INVERSE POWER METHOD (Target Scalar s)")
    print("=" * 60)

    A_shift_test = np.array([
        [4, 1, 0, 0],
        [1, 3, 1, 0],
        [0, 1, 2, 1],
        [0, 0, 1, 1]
    ], dtype=float)
    x0_shift = np.array([1, 1, 1, 1], dtype=float)

    target_s = 2.8
    closest_eval, closest_evec, _, _ = shifted_inverse_power_trj(
        A_shift_test, x0_shift, target_shift=target_s, hand_written=True
    )

    all_evals = np.linalg.eig(A_shift_test)[0]
    expected_closest = all_evals[np.argmin(np.abs(all_evals - target_s))]

    print(f"Target Scalar s:                   {target_s}")
    print(f"Eigenvalue closest to s (Shifted):  {round(closest_eval, 6)}")
    print(f"Expected closest (np.linalg.eig):   {round(expected_closest, 6)}")
    print("Eigenvector:\n", np.round(closest_evec, 6))
    print("=" * 60)
