import numpy as np

# print(np.__version__)

# =====================START=====================


def gauss_jordan_solver(A, B, hand_written=True):
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

def lu_solve_multiple_rhs(L, U, rhs_list, hand_written=True, verbose=False):
    """
    Solves A x = b for SEVERAL right-hand sides using ONE factorization.

    This is the C2 question turned into a reusable function, and the whole
    reason LU exists as a separate method from Gauss elimination.

    THEORETICAL COST ARGUMENT:
    - L and U depend only on A, never on b. Factor once: ~(2/3) n^3 flops.
    - Each extra right-hand side is then just forward + backward substitution:
      ~2 n^2 flops. So m right-hand sides cost O(n^3) + m*O(n^2) instead of
      m*O(n^3) if you re-ran Gauss elimination per b.
    - The identity's columns are a special case of "many right-hand sides",
      which is exactly what lu_inverse() above does.

    Returns:
        solutions (list of ndarray): One solution vector per right-hand side.
        P, L, U: The single factorization that was reused for all of them.
    """
    # A = np.asarray(A, dtype=float)
    # # --- FACTOR ONCE (the expensive O(n^3) part) ---
    # L, U = lu_naive(A, hand_written=hand_written, verbose=verbose)

    solutions = []
    for index, b in enumerate(rhs_list):
        b = np.asarray(b, dtype=float)
        # --- PER RHS: only the two cheap O(n^2) triangular solves ---
        y = forward_substitution(L, b, hand_written=hand_written)
        x = backward_substitution(U, y, hand_written=hand_written)
        solutions.append(x)
        if verbose:
            print(f"  rhs #{index + 1}: z = {np.round(y, 6)}  ->  x = {np.round(x, 6)}")

    return solutions, L, U

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
    L, U = lu_naive(A, hand_written=hand_written, verbose=verbose)
    # Note: P @ b permutes vector b according to row swaps stored in P
    z = forward_substitution(L, b, hand_written=hand_written)
    x = backward_substitution(U, z, hand_written=hand_written)
    return x, L, U


A = np.array(
    [
        [2,1,1],
        [4,3,3],
        [2,2,3]
    ], dtype=float
)

B_multi = np.array([
        [3, 7, 4],
        [3, 7, 5],
        [2, 6, 5]
    ], dtype=float)

X_GJ, _ = gauss_jordan_solver(A, B_multi.T, hand_written=True)
print("Multi-RHS Solutions [x1 | x2]:\n", np.round(X_GJ, 6))

L, U = lu_naive(A, hand_written=True)
rhs = [np.array([3, 7, 4], float), np.array([3, 7, 5], float),
           np.array([2, 6, 5], float)]
X_LU ,_,_ = lu_solve_multiple_rhs(L, U, rhs, hand_written=True)
print("Multi-RHS Solutions [x1 | x2]:\n", np.round(X_LU, 6))

print(np.allclose(A@X_GJ, B_multi.T))
print(np.allclose(A@X_LU, B_multi.T))
print(np.allclose(X_LU, X_GJ))

b = np.array([4, 10, 7], dtype=float)

x_g,_ = gauss_jordan_solver(A, b, hand_written=True)
print(x_g)
x_a,_,_ = solve_via_lu(A, b, hand_written=True)
print(x_a)

print(np.allclose(A@x_a, b.T))
print(np.allclose(A@x_g, b.T))













