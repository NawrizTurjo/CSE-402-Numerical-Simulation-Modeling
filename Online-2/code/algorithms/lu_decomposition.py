"""
LU Decomposition (Doolittle Method) -- Canonical Library Functions & Theoretical Mappings
========================================================================================

THEORETICAL ALGORITHM MAPPING:
------------------------------
Factorizes Matrix A (n x n) into Lower Triangular L and Upper Triangular U:
  A = L * U    (Naive / No Pivoting)
  P * A = L * U (With Partial Pivoting, P is Permutation Matrix)

1. DOOLITTLE FACTORIZATION STRUCTURE:
   - L is Unit Lower Triangular: L[i, i] = 1.0, L[i, j] = 0 for j > i.
   - U is Upper Triangular:      U[i, j] = 0 for i > j.
   - Multipliers m_{i,k} = u_{i,k} / u_{k,k} are saved directly into L[i, k].

2. PARTIAL PIVOTING IN LU (P * A = L * U):
   - Pivot Selection: p = argmax_{i >= k} |U[i, k]|.
   - Swaps in U: Swap U[k, :] <-> U[p, :].
   - Swaps in P: Swap P[k, :] <-> P[p, :].
   - Multiplier Swaps in L: Swap previously stored multipliers L[k, :k] <-> L[p, :k].
     *Why?* The permutation P reorganizes equations before factorization. Swap in L ensures
     multipliers correspond to the physically swapped rows!

3. TWO-STEP TRIANGULAR SOLVE (L * z = P * b, U * x = z):
   Given system A * x = b -> P * A * x = P * b -> L * (U * x) = P * b:
   - Step 1 (Forward Substitution): Solves L * z = P * b for z.
     z_i = ((P * b)_i - sum_{j=0}^{i-1} L[i, j] * z_j) / L[i, i]   (L[i, i] = 1)
   - Step 2 (Backward Substitution): Solves U * x = z for x.
     x_i = (z_i - sum_{j=i+1}^{n-1} U[i, j] * x_j) / U[i, i]

4. COMPUTATIONAL FLOP ADVANTAGE:
   - LU Factorization (PA = LU): ~ (2/3) * n^3 flops (Computed ONCE).
   - Solves for each RHS b:      ~ 2 * n^2 flops (Forward + Backward substitution).
   - For m right-hand sides, total work is O(n^3) + m * O(n^2), compared to
     m * O(n^3) if using full Gauss elimination repeatedly!
"""
import numpy as np


def lu_naive(A, hand_written=True, verbose=False):
    """
    Doolittle LU decomposition without pivoting: A = L @ U.

    THEORETICAL STEPS:
    1. Initialize L = Identity matrix (n x n) and U = copy of A.
    2. Compute multiplier m_{i,k} = U[i, k] / U[k, k].
    3. Store m_{i,k} into lower triangular matrix L[i, k].
    4. Eliminate column k in row i of upper matrix U: U[i, k:] -= m * U[k, k:].

    Returns:
        L (ndarray): Unit lower-triangular matrix.
        U (ndarray): Upper-triangular matrix.
    """
    A = A.astype(float).copy()
    n = A.shape[0]
    L = np.eye(n) # <- Identity matrix
    U = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, n):
            # Formula: Multiplier m = U_{i,k} / U_{k,k}
            m = U[i, k] / U[k, k]
            # Store multiplier into lower triangular matrix L
            L[i, k] = m
            # Eliminate row i in U
            if hand_written:
                for j in range(k, n):
                    U[i, j] -= m * U[k, j]
            else:
                U[i, k:] -= m * U[k, k:]

        if verbose:
            print(f"after column {k}: L=\n{np.round(L,4)}\nU=\n{np.round(U,4)}")

    return L, U


def plu_decomposition(A, hand_written=True, verbose=False):
    """
    Doolittle LU decomposition with partial pivoting: P @ A = L @ U.

    THEORETICAL STEPS:
    1. Initialize U = copy(A), L = Identity(n), P = Identity(n).
    2. Pivot: Find row p >= k with max magnitude |U[p, k]|.
    3. Swap Row k <-> Row p in U and P.
    4. CRITICAL: Swap previously computed multipliers L[k, :k] <-> L[p, :k].
    5. Compute multipliers L[i, k] = U[i, k] / U[k, k] and update U[i, k:].

    Returns:
        P (ndarray): Permutation matrix (n x n).
        L (ndarray): Unit lower-triangular matrix (n x n).
        U (ndarray): Upper-triangular matrix (n x n).
        swap_count (int): Total number of row swaps performed.
    """
    A = A.astype(float).copy()
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    swap_count = 0

    for k in range(n - 1):
        # --- PARTIAL PIVOTING: Find p >= k maximizing |U[p, k]| ---
        if hand_written:
            p = k
            for i in range(k + 1, n):
                if abs(U[i, k]) > abs(U[p, k]):
                    p = i
        else:
            p = k + int(np.argmax(np.abs(U[k:, k])))

        if verbose:
            print(f"column {k}: pivot = {U[p, k]:.6g} (row {p})")

        # --- ROW SWAP IN U, P, AND PREVIOUS MULTIPLIERS IN L ---
        if p != k:
            if verbose:
                print(f"  swap: R{k} <-> R{p}")
            if hand_written:
                for j in range(n):
                    U[k, j], U[p, j] = U[p, j], U[k, j]
                    P[k, j], P[p, j] = P[p, j], P[k, j]
                if k > 0:
                    for j in range(k):
                        L[k, j], L[p, j] = L[p, j], L[k, j]
            else:
                U[[k, p]] = U[[p, k]]
                P[[k, p]] = P[[p, k]]
                # Multiplier swap: Swap already stored multipliers in L for columns < k
                if k > 0:
                    L[[k, p], :k] = L[[p, k], :k]
            swap_count += 1

        # Check for zero pivot
        if abs(U[k, k]) < 1e-12:
            continue

        # --- ELIMINATION AND MULTIPLIER STORAGE ---
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            if hand_written:
                for j in range(k, n):
                    U[i, j] -= m * U[k, j]
            else:
                U[i, k:] -= m * U[k, k:]

    return P, L, U, swap_count


def forward_substitution(L, b, hand_written=True):
    """
    Solves L * z = b for unit lower-triangular L (L[i, i] = 1.0).

    THEORETICAL FORMULA:
    z_i = (b_i - sum_{j=0}^{i-1} L[i, j] * z_j) / L[i, i]
    iterating forward from i = 0 to n-1.
    """
    n = len(b)
    z = np.zeros(n)
    for i in range(n):
        if hand_written:
            total_sum = 0.0
            for j in range(i):
                total_sum += L[i, j] * z[j]
            z[i] = (b[i] - total_sum) / L[i, i]
        else:
            z[i] = (b[i] - L[i, :i] @ z[:i]) / L[i, i]
    return z


def backward_substitution(U, z, hand_written=True):
    """
    Solves U * x = z for upper-triangular U.

    THEORETICAL FORMULA:
    x_i = (z_i - sum_{j=i+1}^{n-1} U[i, j] * x_j) / U[i, i]
    iterating backward from i = n-1 down to 0.
    """
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


def solve_via_lu(A, b, hand_written=True, verbose=False):
    """
    Solves linear system A * x = b via PA = LU decomposition.

    THEORETICAL SOLVE STAGES:
    1. Factorize: P * A = L * U
    2. Forward substitution: Solve L * z = P * b for z. (Note: RHS is P @ b)
    3. Backward substitution: Solve U * x = z for x.

    Returns:
        x, P, L, U
    """
    P, L, U, _ = plu_decomposition(A, hand_written=hand_written, verbose=verbose)
    # Note: P @ b permutes vector b according to row swaps stored in P
    z = forward_substitution(L, P @ b, hand_written=hand_written)
    x = backward_substitution(U, z, hand_written=hand_written)
    return x, P, L, U


def lu_determinant(A, hand_written=True):
    """
    Computes det(A) using LU factorization det(P * A) = det(L) * det(U).

    THEORETICAL FORMULA:
    det(P) = (-1)^(number_of_swaps)
    det(L) = 1.0 (since diagonal of unit lower triangular is all 1s)
    det(U) = prod(diag(U))
    det(A) = (-1)^(swaps) * prod(diag(U))
    """
    P, L, U, swaps = plu_decomposition(A, hand_written=hand_written)
    if hand_written:
        det_val = 1.0
        for i in range(U.shape[0]):
            det_val *= U[i, i]
        det = ((-1) ** swaps) * det_val
    else:
        det = ((-1) ** swaps) * np.prod(np.diag(U))
    return det, P, L, U, swaps


def lu_inverse(A, hand_written=True):
    """
    Computes matrix inverse A^(-1) column by column using one LU factorization.

    THEORETICAL FORMULA:
    A * x_j = e_j  for j = 0 ... n-1
    where e_j is the j-th column of the identity matrix I_n.
    Column j of A^(-1) is the solution vector x_j.
    """
    n = A.shape[0]
    P, L, U, _ = plu_decomposition(A, hand_written=hand_written)
    Ainv = np.zeros((n, n))

    for j in range(n):
        # Create standard basis vector e_j = [0, ..., 1, ..., 0]^T
        e = np.zeros(n)
        e[j] = 1.0
        # Solve A * x_j = e_j via cheap forward & backward substitution
        z = forward_substitution(L, P @ e, hand_written=hand_written)
        Ainv[:, j] = backward_substitution(U, z, hand_written=hand_written)

    return Ainv, P, L, U


def _self_test():
    """Numerical verification of LU functions."""
    A = np.array([[0, 2, 1], [1, 1, 1], [2, 1, 3]], dtype=float)
    b = np.array([5, 3, 6], dtype=float)
    x, P, L, U = solve_via_lu(A, b)
    assert np.allclose(x, np.linalg.solve(A, b)), x
    assert np.linalg.norm(P @ A - L @ U) < 1e-10

    A2 = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
    det, *_ = lu_determinant(A2)
    assert abs(det - np.linalg.det(A2)) < 1e-8

    Ainv, *_ = lu_inverse(A2)
    assert np.linalg.norm(A2 @ Ainv - np.eye(3)) < 1e-8

    print("lu_decomposition.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
