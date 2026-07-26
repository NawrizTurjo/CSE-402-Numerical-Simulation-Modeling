import numpy as np

def lu_decompose(A):
    U = A.astype(float).copy()
    n = len(A)
    L = np.eye(n)
    
    for i in range(n-1):
        for k in range(i+1,n):
            m = U[k][i]/U[i][i]
            L[k][i] = m
            for j in range(i,n):
                U[k][j] = U[k][j]-m*U[i][j]
        print(L)
        print(U)

    return L, U

def lu_decompose_partial_pivot(A):
    U = A.astype(float).copy()
    n = len(A)

    L = np.eye(n)
    P = np.eye(n)

    for i in range(n - 1):

        # Partial Pivoting
        max_row = i
        for k in range(i + 1, n):
            if abs(U[k][i]) > abs(U[max_row][i]):
                max_row = k

        if max_row != i:

            # Swap rows in U
            U[[i, max_row]] = U[[max_row, i]]

            # Swap rows in P
            P[[i, max_row]] = P[[max_row, i]]

            # Swap only previously computed columns of L
            for j in range(i):
                L[i][j], L[max_row][j] = L[max_row][j], L[i][j]

            print(f"Swap R{i+1} <-> R{max_row+1}")

        # Elimination
        for k in range(i + 1, n):

            factor = U[k][i] / U[i][i]

            L[k][i] = factor

            for j in range(i, n):
                U[k][j] = U[k][j] - factor * U[i][j]

        print("\nL =")
        print(L)

        print("U =")
        print(U)

    return P, L, U

def lusolve(L, U, b):
    # LZ = b
    n = len(b)
    Z = np.zeros(n)

    for i in range(n):
        Z[i] = b[i]
        for j in range(i):
            Z[i] = Z[i] - L[i][j]*Z[j]
    
    x = np.zeros(n)

    for i in range(n-1,-1,-1):
        x[i] = Z[i]
        for j in range(i+1,n):
            x[i] = x[i] - U[i][j]*x[j]
        x[i] = x[i]/U[i][i]
    
    return x


A = np.array([[25,5,1],[64,8,1],[144,12,1]])
b = np.array([106.8,177.2,279.2])
L, U =lu_decompose(A)

x = lusolve(L, U, b)

print()
print(x)

print(L@U)

P, L, U = lu_decompose(A)

# Solve PAx = Pb
b2 = P @ b

print(np.linalg.norm(A @ x - b))