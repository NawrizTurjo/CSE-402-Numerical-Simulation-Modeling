import numpy as np

def invpowermethod(mat, x, iter):
    # to find smallest eig value
    if iter <= 0:
        return
    def lu_decomp(mat):
        no_of_var = len(mat[0])
        no_of_eqn = len(mat)

        l = np.identity(no_of_eqn)

        for i in range(0, no_of_var):
            pivot = mat[i][i]
            for j in range(i+1, no_of_eqn):
                multiplier = mat[j][i] / pivot
                l[j][i] = multiplier
                for k in range(no_of_var):
                    mat[j][k] = mat[j][k]- multiplier * mat[i][k]

        u = mat.copy()
        return l.tolist(), u

    def solve(eqns):
        #thru fwd sub
        vars = len(eqns[0])-1
        neqn = len(eqns)
        eqns = np.array(eqns)
        tmp = np.array_split(eqns, [vars], axis=1) #[vars] sends the index from where to split, vars would otherwise just note the number of groups
        a,c = tmp[0], tmp[1]
        l, u = lu_decomp(a)
        # solve lz = c
        #fwd sub
        z =  []
        for i in range(0, vars):
            res = c[i]
            for j in range(0, i):
                res = res - l[i][j] * z[j]
            z.append(res)

        # solve ux = z
        # backsub
        x = [0] * vars
        for i in range(vars-1, -1, -1):
            res = z[i]
            for j in range(i+1, vars):
                res = res - u[i][j] * x[j]
            res = res/u[i][i]
            x[i] = res
        return x
    # print(mat, x)
    augmat = np.column_stack((mat, x))
    y = solve(augmat)
    eig = np.max(np.abs(y))
    y = y/eig
    print(y)
    iter -=1
    print(iter)
    return 1/eig, invpowermethod(mat, y, iter)

mat = [[8, 2, 0, 0],
       [2, 8, 0, 0],
       [0, 0, 3, 1],
       [0, 0, 1, 3]]

print(invpowermethod(mat, [1,0,1,0], 10))

print(np.matmul(np.linalg.matrix_power(mat, -1), [1,0,1,0]))

