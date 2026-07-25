import numpy as np

def gauss_jordan(A, b):
    """
    Solves Ax = b using Gauss-Jordan Elimination.
    Drives the coefficient matrix all the way to the Identity matrix.
    """
    n = A.shape[0]
    Ab = np.hstack([A, b.reshape(-1, 1)]).astype(float)
    
    print("\nInitial Augmented Matrix:")
    print(Ab)
    
    for k in range(n):
        print(f"\n--- Processing Column {k+1} ---")
        # Partial Pivoting
        max_idx = k + np.argmax(np.abs(Ab[k:, k]))
        if max_idx != k:
            print(f"Swapping row {k+1} and row {max_idx+1}")
            Ab[[k, max_idx]] = Ab[[max_idx, k]]
            
        pivot = Ab[k, k]
        if np.abs(pivot) < 1e-12:
            raise ValueError("Matrix is singular")
            
        # Normalize the pivot row so the pivot becomes 1
        print(f"Normalizing row {k+1} by its pivot {pivot:.4f}")
        Ab[k, :] /= pivot
        
        # Eliminate the variable from EVERY OTHER equation (above and below)
        for i in range(n):
            if i != k:
                factor = Ab[i, k]
                print(f"Eliminating row {i+1} using factor {factor:.4f}")
                Ab[i, :] -= factor * Ab[k, :]
                
        print("Augmented Matrix:")
        print(np.round(Ab, 4))
                
    # The solution is just the last column now
    x = Ab[:, -1]
    
    print("\nFinal Coefficient Matrix is Identity.")
    print("Solution vector x:", np.round(x, 4))
    
    return x

if __name__ == "__main__":
    A = np.array([
        [3, -0.1, -0.2],
        [0.1, 7, -0.3],
        [0.3, -0.2, 10]
    ], dtype=float)
    b = np.array([7.85, -19.3, 71.4], dtype=float)
    
    print("Solving with Gauss-Jordan Elimination:")
    gauss_jordan(A, b)
