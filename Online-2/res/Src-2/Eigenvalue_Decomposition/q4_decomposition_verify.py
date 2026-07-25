import numpy as np

def eigenvalue_decomposition_verify(A):
    """
    Verifies A = V * Lambda * V^{-1} by:
    1. Computing eigenvalues and eigenvectors using NumPy.
    2. Constructing V (eigenvector matrix) and Lambda (diagonal eigenvalue matrix).
    3. Computing V^{-1}.
    4. Reconstructing A and checking if it matches the original.
    """
    evals, evecs = np.linalg.eig(A)

    V = evecs  # Each column is an eigenvector
    Lambda = np.diag(evals)
    V_inv = np.linalg.inv(V)

    print("Eigenvalues:")
    for i, val in enumerate(evals):
        print(f"  λ{i+1} = {val.real:.6f}")

    print("\nV (Eigenvector Matrix, columns are eigenvectors):")
    print(np.round(V.real, 6))

    print("\nΛ (Diagonal Eigenvalue Matrix):")
    print(np.round(Lambda.real, 6))

    print("\nV^{-1} (Inverse of Eigenvector Matrix):")
    print(np.round(V_inv.real, 6))

    # Reconstruct A
    A_reconstructed = V @ Lambda @ V_inv
    print("\nReconstructed A = V * Λ * V^{-1}:")
    print(np.round(A_reconstructed.real, 4))

    print("\nOriginal A:")
    print(np.round(A, 4))

    error = np.linalg.norm(A - A_reconstructed.real)
    print(f"\n||A - V*Λ*V^{{-1}}||_2 = {error:.6e}")
    if error < 1e-8:
        print("Decomposition VERIFIED successfully!")
    else:
        print("Decomposition has significant error.")


if __name__ == "__main__":
    # 2x2 example
    print("=" * 50)
    print("2×2 Matrix")
    print("=" * 50)
    A_2x2 = np.array([
        [4, 1],
        [2, 3]
    ], dtype=float)
    print("Matrix A:")
    print(A_2x2)
    eigenvalue_decomposition_verify(A_2x2)

    # 4x4 example (matching the online question)
    print(f"\n\n{'=' * 50}")
    print("4×4 Matrix")
    print("=" * 50)
    A_4x4 = np.array([
        [4, 1, 0, 0],
        [1, 5, 1, 0],
        [0, 1, 6, 1],
        [0, 0, 1, 7]
    ], dtype=float)
    print("Matrix A:")
    print(A_4x4)
    eigenvalue_decomposition_verify(A_4x4)
