import numpy as np

def power_iteration(A, num_iterations: int = 1000, tolerance: float = 1e-9):
    """
    Computes the dominant eigenvalue and eigenvector of matrix A 
    using the Power Iteration method.
    """
    n = A.shape[0]
    # Start with a random initial vector (or all ones)
    v = np.ones(n)
    
    eigenvalue = 0.0
    for _ in range(num_iterations):
        # Multiply by A
        v_next = np.dot(A, v)
        
        # Calculate the Rayleigh quotient to estimate the eigenvalue
        eigenvalue_next = np.dot(v.T, v_next) / np.dot(v.T, v)
        
        # Normalize the vector (L2 norm) to prevent overflow/underflow
        norm = np.linalg.norm(v_next)
        v_next = v_next / norm
        
        # Check convergence
        if np.abs(eigenvalue - eigenvalue_next) < tolerance:
            eigenvalue = eigenvalue_next
            v = v_next
            break
            
        eigenvalue = eigenvalue_next
        v = v_next

    return eigenvalue, v

def verify_with_numpy(A, custom_eval, custom_evec):
    """
    Verifies the computed eigenvalue and eigenvector using NumPy's linalg.eig.
    """
    evals, evecs = np.linalg.eig(A)
    
    # Find the dominant eigenvalue from NumPy
    max_idx = np.argmax(np.abs(evals))
    np_dom_eval = evals[max_idx]
    np_dom_evec = evecs[:, max_idx]
    
    print("\n--- Verification ---")
    print(f"Custom Eigenvalue: {custom_eval:.6f}")
    print(f"NumPy  Eigenvalue: {np_dom_eval:.6f}")
    
    print(f"\nCustom Eigenvector: {np.round(custom_evec, 6)}")
    print(f"NumPy  Eigenvector: {np.round(np_dom_evec, 6)}")
    
    # Check if they match (accounting for potential sign flip)
    dot_product = np.dot(custom_evec, np_dom_evec)
    if np.isclose(np.abs(dot_product), 1.0, atol=1e-5):
        print("\nVerification SUCCESS: The eigenvectors match (or are valid scalar multiples/flipped signs).")
    else:
        print("\nVerification FAILED: The eigenvectors do not match.")

if __name__ == "__main__":
    # Example 4x4 Matrix
    A = np.array([
        [4, 1, 0, 0],
        [1, 5, 1, 0],
        [0, 1, 6, 1],
        [0, 0, 1, 7]
    ], dtype=float)

    print("Matrix A:")
    print(A)

    dom_eval, dom_evec = power_iteration(A)
    verify_with_numpy(A, dom_eval, dom_evec)
