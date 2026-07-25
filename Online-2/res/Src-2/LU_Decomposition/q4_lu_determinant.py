import numpy as np

def determinant_lu(A):
    """
    Calculates the determinant of A using LU decomposition (Gauss Elimination).
    det(A) = (-1)^S * (u_11 * u_22 * ... * u_nn) where S is the number of row swaps.
    """
    n = A.shape[0]
    U = A.copy().astype(float)
    num_swaps = 0
    
    print("\n--- Starting Elimination for Determinant ---")
    for k in range(n - 1):
        max_idx = k + np.argmax(np.abs(U[k:, k]))
        
        if max_idx != k:
            print(f"Swapping row {k+1} and row {max_idx+1}")
            U[[k, max_idx]] = U[[max_idx, k]]
            num_swaps += 1
            
        if np.abs(U[k, k]) < 1e-12:
            print("Zero pivot encountered. Determinant is 0.")
            return 0.0
            
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            U[i, k:] -= factor * U[k, k:]
            
    print("\nUpper Triangular Matrix (U):")
    print(np.round(U, 4))
    
    # Calculate determinant
    det_U = np.prod(np.diag(U))
    det_A = det_U * ((-1) ** num_swaps)
    
    print(f"\nNumber of row swaps: {num_swaps}")
    print(f"Product of U's diagonals: {det_U:.4f}")
    print(f"Determinant of A: {det_A:.4f}")
    
    return det_A

if __name__ == "__main__":
    A = np.array([
        [0, 2, 1],
        [1, -2, -3],
        [-1, 1, 2]
    ], dtype=float)
    
    print("Matrix A:")
    print(A)
    
    det_A = determinant_lu(A)
    
    print(f"\nNumPy Verification: {np.linalg.det(A):.4f}")
