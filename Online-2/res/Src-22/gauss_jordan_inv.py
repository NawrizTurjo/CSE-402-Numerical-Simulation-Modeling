import numpy as np
def gji(A):
    A = A.astype(float).copy()
    n = len(A)
    I = np.eye(n)

    for i in range(n):
        maxr = i
        for j in range(i+1,n):
            if(abs(A[j][i])> abs(A[maxr][i])):
                maxr = j
        A[[i, maxr]] = A[[maxr,i]]
        I[[i, maxr]] = I[[maxr,i]]
        pivot = A[i][i]
        for k in range(n):
            A[i][k] /=pivot
            I[i][k] /=pivot
        
        for k in range(n):
            if k != i:
                factor = A[k][i]
                for j in range(n):
                    A[k][j] = A[k][j] - factor*A[i][j]
                    I[k][j] = I[k][j] - factor*I[i][j]
        print(A)
        print(I)

    return I

n = 3

A = np.array([[3,-.1,-.2],[.1,7,-.2],[.3,-.2,10]])

I = gji(A)
print(I)

print(np.linalg.inv(A))

    
