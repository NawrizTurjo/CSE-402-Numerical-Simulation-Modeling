import numpy as np
np.set_printoptions(suppress=True)

def round_to_n_sig(x, n):
    return round(x, -int(np.floor(np.abs(np.log10(x))))+(n-1))

def power_method(mat, x, iter, prev_eig, prev_vec):
    if iter <= 0: 
        return prev_eig, prev_vec
    iter -=1
    y = np.matmul(mat, x)
    eigi = np.argmax(np.abs(y))
    eig = y[eigi]
    y = y/eig
    return power_method(mat, y, iter, eig, y)

def eigvec(mat, eigval):
    mat = mat.copy()
    for i in range(len(mat)):
        mat[i][i] -= eigval
    return mat
    

def calc_all_eigens(mat):
    eigs = []
    vecs = []
    for i in range(len(mat)):
        eig, vec = power_method(mat, np.random.randint(-10, 10, size=len(mat)), 100, float('inf'), np.random.randint(-10, 10, size=len(mat)))
        eigs.append(eig)
        vec = vec/np.sqrt(np.sum(np.power(vec, 2)))
        mat = mat.copy() - eig * np.outer(vec, vec)
        # remove the contribution of eig from a
        vecs.append(vec)
    return eigs, vecs


        
    pass

mat = [[8, 2, 0, 0],
       [2, 8, 0, 0],
       [0, 0, 3, 1],
       [0, 0, 1, 3]]
k = mat.copy()
print(calc_all_eigens(mat))

# print(eigvec(mat, 100))

print(np.linalg.eig(k))

print(round_to_n_sig(1234.659, 1))