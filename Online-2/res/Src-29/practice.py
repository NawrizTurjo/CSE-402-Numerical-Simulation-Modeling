import numpy as np



def fwd_subs(L,C):
    Z = np.zeros(n)

    for i in range(n):
        total = 0.0
        for j in range(i):
            total += L[i,j]*Z[j]
        Z[i] = (C[i] - total)

    return Z




def gaus_jordan(A,b):
    aug = []
    i=0
    n = len(b)


    #normalize pivot
    aug[i] = aug[i] / aug[i,i]

    for j in range(n):
        if j != i:
            factor = aug[j,i]/aug[i,i]
            aug[j] -= factor * aug[i]

    return aug[:,-1]









# def gauss_elim(A,b):

#     A = np.array(A, dtype=float)
#     b = np.array(b, dtype=float)

#     aug = np.hstack([A, b.reshape(-1,1)])

#     n = len(b)
#     #fwd with pp
#     for i in range(n):
#         max_row = np.argmax(np.abs(aug[i:,i])) + i

#         if np.isclose(aug[max_row,i],0):
#             raise ValueError("xero")
#         if max_row != i:
#             aug[[i, max_row]] = aug[[max_row,i]]

#         for j in range(i+1,n):
#             factor = aug[j,i] / aug[i,i]
#             aug[j,i:] -= factor * aug[i,i:]

#     x = np.zeros(n)
#     for i in range(n-1,-1,-1):
#         x[i] = (aug[i,-1] - np.dot(aug[i,i+1:n], x[i+1:n]))/aug[i,i]

#     return x