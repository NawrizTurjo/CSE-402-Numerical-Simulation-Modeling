import numpy as np

def gauss_jordan(A, b):
    """
    Solve Ax = b using Gauss-Jordan elimination with partial pivoting.
    Reduces [A | b] to [I | x].
    
    Parameters:
        A : (n, n) coefficient matrix
        b : (n,) right-hand side vector
    
    Returns:
        x : (n,) solution vector
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # Build augmented matrix [A | b]
    Aug = np.hstack([A, b.reshape(-1, 1)])

    for i in range(n):
        # Partial pivoting: bring the largest pivot to row i
        max_row = np.argmax(np.abs(Aug[i:, i])) + i
        if np.isclose(Aug[max_row, i], 0):
            raise ValueError("Matrix is singular or nearly singular.")
        if max_row != i:
            Aug[[i, max_row]] = Aug[[max_row, i]]

        # Normalize pivot row so pivot element becomes 1
        Aug[i] = Aug[i] / Aug[i, i]

        # Eliminate all other entries in this column (above AND below)
        for j in range(n):
            if j != i:
                factor = Aug[j, i]
                Aug[j] -= factor * Aug[i]

    # Solution is now the last column
    x = Aug[:, -1]
    return x


def gauss_jordan_inverse(A):
    """
    Compute the inverse of matrix A using Gauss-Jordan elimination,
    by reducing [A | I] to [I | A^-1].
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # Build augmented matrix [A | I]
    Aug = np.hstack([A, np.eye(n)])

    for i in range(n):
        max_row = np.argmax(np.abs(Aug[i:, i])) + i
        if np.isclose(Aug[max_row, i], 0):
            raise ValueError("Matrix is singular; inverse does not exist.")
        if max_row != i:
            Aug[[i, max_row]] = Aug[[max_row, i]]

        Aug[i] = Aug[i] / Aug[i, i]

        for j in range(n):
            if j != i:
                factor = Aug[j, i]
                Aug[j] -= factor * Aug[i]

    A_inv = Aug[:, n:]
    return A_inv


# Example usage
if __name__ == "__main__":
    A = [[2, 1, -1],
         [-3, -1, 2],
         [-2, 1, 2]]
    b = [8, -11, -3]

    x = gauss_jordan(A, b)
    print("Solution:", x)
    print("Check (np.linalg.solve):", np.linalg.solve(A, b))

    A_inv = gauss_jordan_inverse(A)
    print("\nInverse of A:\n", A_inv)
    print("Check (np.linalg.inv):\n", np.linalg.inv(A))