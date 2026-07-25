import numpy as np

def power_iteration(A, tol=1e-9, max_iter=1000):
    """
    Finds the dominant eigenvalue and its corresponding eigenvector
    using the Power Iteration Method.
    """
    n = A.shape[0]
    x = np.ones(n)  # Initial guess

    eigenvalue = 0.0
    print("\n--- Power Iteration ---")
    for iteration in range(1, max_iter + 1):
        # Multiply by A
        y = np.dot(A, x)

        # Find the entry with the largest absolute value (eigenvalue estimate)
        eigenvalue_new = y[np.argmax(np.abs(y))]

        # Normalize
        x_new = y / eigenvalue_new

        # Check convergence
        if np.abs(eigenvalue_new - eigenvalue) < tol:
            print(f"Converged after {iteration} iterations.")
            eigenvalue = eigenvalue_new
            x = x_new
            break

        eigenvalue = eigenvalue_new
        x = x_new

    # Normalize the eigenvector to unit length (L2 norm) before returning
    eigenvector = x / np.linalg.norm(x)

    return eigenvalue, eigenvector


def verify_with_numpy(A, custom_eval, custom_evec):
    """Verifies the result against NumPy's np.linalg.eig."""
    evals, evecs = np.linalg.eig(A)

    # Find the dominant eigenvalue from NumPy
    max_idx = np.argmax(np.abs(evals))
    np_eval = evals[max_idx].real
    np_evec = evecs[:, max_idx].real

    print("\n--- Verification with NumPy ---")
    print(f"Custom  Eigenvalue : {custom_eval:.6f}")
    print(f"NumPy   Eigenvalue : {np_eval:.6f}")
    print(f"Match: {np.isclose(custom_eval, np_eval, atol=1e-4)}")

    print(f"\nCustom  Eigenvector: {np.round(custom_evec, 6)}")
    print(f"NumPy   Eigenvector: {np.round(np_evec, 6)}")

    # Eigenvectors may differ by sign (v and -v are both valid)
    dot = np.abs(np.dot(custom_evec, np_evec))
    print(f"Dot product (should be ~1.0): {dot:.6f}")
    if np.isclose(dot, 1.0, atol=1e-4):
        print("Eigenvectors MATCH (sign difference is expected and acceptable).")
    else:
        print("Eigenvectors DO NOT match.")


if __name__ == "__main__":
    A = np.array([
        [4, 1, 0, 0],
        [1, 5, 1, 0],
        [0, 1, 6, 1],
        [0, 0, 1, 7]
    ], dtype=float)

    print("Matrix A:")
    print(A)

    dom_eval, dom_evec = power_iteration(A)

    print(f"\nDominant Eigenvalue : {dom_eval:.6f}")
    print(f"Eigenvector (normalized): {np.round(dom_evec, 6)}")

    verify_with_numpy(A, dom_eval, dom_evec)
