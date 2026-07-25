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


def matrix_power(mat, n):
    # first eig decomp
    # then vAv-1
    # then A^n
    eigs, vecs = calc_all_eigens(mat)
    v = np.column_stack(vecs)
    vinv = np.linalg.inv(v)
    A = np.zeros(shape=(len(eigs), len(eigs)))
    for i in range(len(eigs)):
        A[i][i] = eigs[i]

    for i in range(len(A)):
        A[i][i] = np.power(A[i][i], n)
    
    return np.matmul(np.matmul(v, A), vinv)


a = [[2, 1],
     [1, 2]]

print(matrix_power(a, 10))
print(np.linalg.matrix_power(a, 10))