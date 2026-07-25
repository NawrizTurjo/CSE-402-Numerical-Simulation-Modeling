import numpy as np

def gauss_elimination(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # Build augmented matrix [A | b]
    Aug = np.hstack([A, b.reshape(-1, 1)])

    # Forward elimination with partial pivoting
    for i in range(n):
        # Partial pivoting: swap rows to put the largest pivot on top
        max_row = np.argmax(np.abs(Aug[i:, i])) + i
        if np.isclose(Aug[max_row, i], 0):
            raise ValueError("Matrix is singular or nearly singular.")
        if max_row != i:
            Aug[[i, max_row]] = Aug[[max_row, i]]

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            factor = Aug[j, i] / Aug[i, i]
            Aug[j, i:] -= factor * Aug[i, i:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Aug[i, -1] - np.dot(Aug[i, i + 1:n], x[i + 1:n])) / Aug[i, i]

    return x


# Example usage
if __name__ == "__main__":
    A = [[2, 1, -1],
         [-3, -1, 2],
         [-2, 1, 2]]
    b = [8, -11, -3]

    x = gauss_elimination(A, b)
    print("Solution:", x)
