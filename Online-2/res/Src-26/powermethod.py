import numpy as np

def power_method(mat, x, iter):
    if iter <= 0: 
        return
    iter -=1
    y = np.matmul(mat, x)
    eig = np.max(np.abs(y))
    y = y/eig
    return eig, power_method(mat, y, iter)
    pass
    


a = [[2,1],
     [1,2]]

print(power_method(a, [1,0], 10))