import numpy as np
import matplotlib.pyplot as plt

def gauss_jordan_elim(sys_of_eqns):
    no_of_var = len(sys_of_eqns[0])-1
    no_of_eqn = len(sys_of_eqns)
    for i in range(no_of_var):

        # normalize
        for k in range (no_of_var+1):
            if i == k:
                continue
            sys_of_eqns[i][k] /= sys_of_eqns[i][i]
        sys_of_eqns[i][i] = 1
        # print(sys_of_eqns, i, sys_of_eqns[i][i])
        for j in range(no_of_eqn):
            if j == i:
                continue
            multiplier = sys_of_eqns[j][i]
            for k in range(no_of_var+1):
                sys_of_eqns[j][k] -= multiplier * sys_of_eqns[i][k]

    return sys_of_eqns


system = [[2, 1, 3],
          [1, -1, 0]]

print(gauss_jordan_elim(system))