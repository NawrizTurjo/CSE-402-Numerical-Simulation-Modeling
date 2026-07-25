import numpy as np

def lu_decomposition(A):

    A = np.array(A, dtype=float)
    n = A.shape[0]

    U = A.copy()          # U starts as a copy of A, gets eliminated in-place
    L = np.eye(n)          # L starts as identity, gets multipliers filled in

    for i in range(n):                      # pivot row
        if np.isclose(U[i, i], 0):
            raise ValueError("Zero pivot encountered; pivoting required.")

        for j in range(i + 1, n):            # rows below pivot
            factor = U[j, i] / U[i, i]
            L[j, i] = factor                 # store multiplier in L

            U[j, i:] -= factor * U[i, i:]

    return L, U


def forward_substitution(L, C):
    """Solve LZ = C for Z, where L is lower triangular (unit diagonal), using loops."""
    n = len(C)
    Z = np.zeros(n)

    for i in range(n):
        total = 0.0
        for j in range(i):
            total += L[i, j] * Z[j]
        Z[i] = (C[i] - total)
    return Z


def back_substitution(U, Z):
    """Solve UX = Z for X, where U is upper triangular, using loops."""
    n = len(Z)
    X = np.zeros(n)

    for i in range(n - 1, -1, -1):
        total = 0.0
        for j in range(i + 1, n):
            total += U[i, j] * X[j]
        X[i] = (Z[i] - total) / U[i, i]

    return X


def lu_solve(A, C):
    """
    Solve AX = C using LU decomposition.
    Steps: A = LU  ->  LZ = C (forward sub)  ->  UX = Z (back sub)
    """
    C = np.array(C, dtype=float)
    L, U = lu_decomposition(A)
    Z = forward_substitution(L, C)
    X = back_substitution(U, Z)
    return X, L, U


def lu_inverse(A):
    """
    Compute the inverse of A using LU decomposition, solving AX = I
    one column at a time. L, U computed only once.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]

    L, U = lu_decomposition(A)

    I = np.eye(n)
    A_inv = np.zeros((n, n))

    for col in range(n):
        e_col = I[:, col]
        Z = forward_substitution(L, e_col)
        X = back_substitution(U, Z)
        A_inv[:, col] = X

    return A_inv, L, U


# Example usage
if __name__ == "__main__":
    A = [[2, 1, -1],
         [-3, -1, 2],
         [-2, 1, 2]]
    C = [8, -11, -3]

    # Solve AX = C
    X, L, U = lu_solve(A, C)
    print("Solution X:", X)
    print("Check (np.linalg.solve):", np.linalg.solve(A, C))

    print("\nL =\n", L)
    print("\nU =\n", U)
    print("\nCheck L @ U =\n", L @ U)
    print("Original A =\n", np.array(A, dtype=float))

    # Compute inverse using the SAME L, U machinery
    A_inv, L, U = lu_inverse(A)
    print("\nInverse of A:\n", A_inv)
    print("\nCheck (np.linalg.inv):\n", np.linalg.inv(A))
    print("\nCheck A @ A_inv =\n", np.array(A, dtype=float) @ A_inv)