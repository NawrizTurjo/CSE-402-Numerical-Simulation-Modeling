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

def power_method(A, tol = 1e-16):
    A = A.astype(float).copy()
    n = len(A)
    x = np.array([1,2,3,4])
    iter = 1
    prev = None
    while True:
        y = A@x
        eigenval = np.max(abs(y))
        x = y/eigenval
        if iter == 1:
            error = None
        else: 
            error = abs((eigenval-prev)/eigenval)

            if error < tol:
                break
        iter +=1
        prev = eigenval
    return eigenval, x

def smalpow(A, tol = 1e-16):
    A = A.astype(float).copy()
    n = len(A)
    x = np.array([1,2,3,4])
    iter = 1
    prev_eigen = None
    L, U = lu_decompose(A)
    while True:
        y = lusolve(L, U, x)
        mu = np.max(abs(y))
        x = y/mu
        eigenval = 1/mu
        if iter != 1:
            err = abs((eigenval-prev_eigen)/eigenval)
            if err < tol:
                break

        iter +=1
        prev_eigen = eigenval
    return eigenval, x

def deflation(A, eig, v):
    v1 = v/np.linalg.norm(v)
    A2 = A - eig*np.outer(v1,v1)
    return A2

A = np.array([[8,2,0,0],[2,8,0,0],[0,0,3,1],[0,0,1,3]])
eig1 , v1 = power_method(A)
print(eig1)
print(v1)
# v1 = v1/np.linalg.norm(v1)
A2 = deflation(A,eig1,v1)
eig2, v2 = power_method(A2)
print(eig2)
print(v2)
# sei, xs = smalpow(A)
# print(sei)
# print(xs)
print(np.linalg.eigvals(A))
