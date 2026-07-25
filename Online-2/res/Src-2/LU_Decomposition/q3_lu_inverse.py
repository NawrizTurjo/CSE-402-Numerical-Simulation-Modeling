import numpy as np
from LU_Decomposition.q2_lu_decomposition import plu_decomposition, forward_substitution, backward_substitution

def invert_matrix_lu(A):
    """
    Finds the inverse of matrix A by solving Ax = e_j for every column 
    of the identity matrix, using LU decomposition.
    """
    n = A.shape[0]
    print("\nDecomposing matrix A into PA = LU...")
    P, L, U = plu_decomposition(A)
    
    A_inv = np.zeros((n, n))
    I = np.eye(n)
    
    print("\nSolving LUx = e_j for each column of the identity matrix...")
    for j in range(n):
        e_j = I[:, j]
        # Multiply by P since PAx = e_j -> LUx = Pe_j
        Pe_j = np.dot(P, e_j)
        
        # Solve Lz = Pe_j
        z = forward_substitution(L, Pe_j)
        
        # Solve Ux_j = z
        x_j = backward_substitution(U, z)
        
        # The solution x_j forms the j-th column of A_inv
        A_inv[:, j] = x_j
        print(f"Column {j+1} of A_inv: {np.round(x_j, 4)}")
        
    return A_inv

if __name__ == "__main__":
    A = np.array([
        [4, 3, -1],
        [-2, -4, 5],
        [1, 2, 6]
    ], dtype=float)
    
    print("Original Matrix A:")
    print(A)
    
    A_inv = invert_matrix_lu(A)
    
    print("\nCalculated Inverse A_inv:")
    print(np.round(A_inv, 4))
    
    print("\nVerification (A * A_inv should be Identity):")
    I_approx = np.dot(A, A_inv)
    print(np.round(I_approx, 4))
