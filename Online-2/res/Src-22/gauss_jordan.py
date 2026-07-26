import numpy as np

def gauss_jordan(A,b):
    A= A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    for i in range(n):
        max_row = i
        for j in range(i+1,n):
            if(abs(A[j][i]) > abs(A[max_row][i])):
                max_row = j

        A[[i, max_row]] = A[[max_row,i]]
        b[i],b[max_row] = b[max_row], b[i]

        pivot = A[i][i]

        for j in range(n):
            A[i][j] /=pivot
        b[i] /=pivot

        for k in range(n):
            if(k != i):
                factor = A[k][i]
                for j in range(n):
                    A[k][j] = A[k][j] - factor*A[i][j]
                b[k] = b[k] -factor*b[i]
        print(A)
        print(b)

    return b


n = 3 #int(input())

A = np.array([[3,-.1,-.2],[.1,7,-.2],[.3,-.2,10]])
b = np.array([7.85,-19.3,71.4])

x = gauss_jordan(A,b)
print(x)
print(np.linalg.solve(A,b))