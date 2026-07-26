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

def inv_lu(A):
    n = len(A)

    L, U = lu_decompose(A)
    inverse = np.zeros((n,n))

    for i in range(n):
        b = np.zeros(n)
        b[i] = 1
        x = lusolve(L,U,b)
        inverse[:, i] = x

    return inverse

A = np.array([[4, 3],
              [6, 3]], dtype=float)
inv = inv_lu(A)

print(inv)
print(np.linalg.inv(A))

