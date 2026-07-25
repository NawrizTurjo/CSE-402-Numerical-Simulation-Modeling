import numpy as np
import matplotlib.pyplot as plt

def gauss_elimination(eqns):
    no_of_var_and_c = len(eqns[0])
    no_of_eqn = len(eqns)
    for i in range(no_of_var_and_c-1):
        pivot = eqns[i][i] 
        for j in range(i+1, no_of_eqn):
            # cancel a_{jk} for all j>i
            multiplier = eqns[j][i] / pivot
            for k in range(no_of_var_and_c):
                eqns[j][k] -= multiplier * eqns[i][k]
            #print system after update

    return eqns

def gauss_elim_with_pp(eqns):
    pass

def backward_sub(upper_triangular):
    no_of_var_and_c = len(upper_triangular[0])
    res = [0] * (no_of_var_and_c - 1)
    for i in range(no_of_var_and_c-1-1, -1, -1):
        # ith row er col >= i entry gula non zero oigula ke sub
        for j in range(i, no_of_var_and_c-1):
            upper_triangular[i][no_of_var_and_c-1] -= upper_triangular[i][j] * res[j]
        res[i] = upper_triangular[i][no_of_var_and_c-1]/upper_triangular[i][i]
    
    return res



system = [[2, 1, 3],
          [1, -1, 0]]

print(backward_sub(upper_triangular=gauss_elimination(system)))
