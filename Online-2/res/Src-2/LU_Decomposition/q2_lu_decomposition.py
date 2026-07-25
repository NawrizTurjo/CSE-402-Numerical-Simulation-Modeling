import numpy as np

def plu_decomposition(A):
    """
    Performs LU Decomposition with Partial Pivoting (PA = LU).
    """
    n = A.shape[0]
    U = A.copy().astype(float)
    L = np.eye(n, dtype=float)
    P = np.eye(n, dtype=float)
    
    print("\n--- Starting PLU Decomposition ---")
    for k in range(n - 1):
        # Find the row with the largest absolute value in the current column
        max_idx = k + np.argmax(np.abs(U[k:, k]))
        
        if max_idx != k:
            print(f"Swapping row {k+1} and row {max_idx+1}")
            # Swap rows in U
            U[[k, max_idx]] = U[[max_idx, k]]
            # Swap rows in P
            P[[k, max_idx]] = P[[max_idx, k]]
            # Swap rows in L (only the multipliers computed so far, before column k)
            if k > 0:
                L[[k, max_idx], :k] = L[[max_idx, k], :k]
                
        if np.abs(U[k, k]) < 1e-12:
            print("Warning: Matrix is singular or nearly singular.")
            continue
            
        # Eliminate entries below the pivot
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor  # Store multiplier in L
            U[i, k:] -= factor * U[k, k:]
            
    return P, L, U

def forward_substitution(L, b):
    """Solves Lz = b for z."""
    n = L.shape[0]
    z = np.zeros(n)
    for i in range(n):
        z[i] = b[i] - np.dot(L[i, :i], z[:i])
    return z

def backward_substitution(U, z):
    """Solves Ux = z for x."""
    n = U.shape[0]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if np.abs(U[i, i]) < 1e-12:
            raise ValueError("Matrix is singular")
        x[i] = (z[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def solve_system_lu(A, b):
    """Solves Ax = b using PA = LU decomposition."""
    P, L, U = plu_decomposition(A)
    
    print("\nP (Permutation Matrix):")
    print(np.round(P, 4))
    print("\nL (Lower Triangular Matrix):")
    print(np.round(L, 4))
    print("\nU (Upper Triangular Matrix):")
    print(np.round(U, 4))
    
    # PAx = Pb -> LUx = Pb
    Pb = np.dot(P, b)
    print(f"\nPermuted b (Pb): {np.round(Pb, 4)}")
    
    # Solve Lz = Pb
    z = forward_substitution(L, Pb)
    print(f"Intermediate vector z (from Lz = Pb): {np.round(z, 4)}")
    
    # Solve Ux = z
    x = backward_substitution(U, z)
    print(f"Final Solution vector x (from Ux = z): {np.round(x, 4)}")
    
    return x

if __name__ == "__main__":
    A = np.array([
        [2, 1, -1],
        [-3, -1, 2],
        [-2, 1, 2]
    ], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    
    print("Matrix A:")
    print(A)
    print("Vector b:", b)
    
    solve_system_lu(A, b)
