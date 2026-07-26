import numpy as np

def gauss_elim(A,b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)
    #fe
    for i in range(n):
        max = abs(A[i][i])
        max_row = i
        for j in range(i+1,n):
            if(abs(A[j][i]) > max):
                max = abs(A[j][i])
                max_row = j
        A[[i,max_row]] = A[[max_row,i]]
        b[i], b[max_row] = b[max_row], b[i]

        for k in range(i+1, n):
            factor = A[k][i]/A[i][i]
            for j in range(i,n):
                A[k][j] = A[k][j] - factor*A[i][j]
        
            b[k] = b[k] - factor*b[i]
            print(A)
            print(b)
    x = np.zeros(n)

    for i in range(n-1,-1,-1):
        x[i] = b[i]
        for j in range(i+1,n):
            x[i] = x[i]-A[i][j]*x[j]
        x[i] = x[i]/A[i][i]
    return x                    

n = 3 #int(input())

# A = np.zeros((n,n))
# b = np.zeros(n)


# for i in range(n):
#     for j in range(n):
#         A[i][j] = float(input())

# for i in range(n):
#     b[i] = float(input())

A = np.array([[25,5,1],[64,8,1],[144,12,1]])
b = np.array([106.8,177.2,279.2])

x = gauss_elim(A,b)
print(x)
y = x = np.linalg.solve(A, b)
print(y)