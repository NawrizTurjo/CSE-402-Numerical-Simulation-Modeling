import numpy as np

def gauss_elim(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()

    n = len(b)
    swap_count = 0

    # Forward Elimination
    for i in range(n):

        max_val = abs(A[i][i])
        max_row = i

        for j in range(i+1, n):
            if abs(A[j][i]) > max_val:
                max_val = abs(A[j][i])
                max_row = j

        # Row swap
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[i], b[max_row] = b[max_row], b[i]
            swap_count += 1

        # Elimination
        for k in range(i+1, n):

            factor = A[k][i] / A[i][i]

            for j in range(i, n):
                A[k][j] -= factor * A[i][j]

            b[k] -= factor * b[i]

        print("A =")
        print(A)
        print("b =", b)
        print()

    # ---------- Determinant ----------
    det = 1

    for i in range(n):
        det *= A[i][i]

    if swap_count % 2 == 1:
        det = -det

    # ---------- Back Substitution ----------
    x = np.zeros(n)

    for i in range(n-1, -1, -1):

        x[i] = b[i]

        for j in range(i+1, n):
            x[i] -= A[i][j] * x[j]

        x[i] /= A[i][i]

    return x, det


A = np.array([[25,5,1],
              [64,8,1],
              [144,12,1]], dtype=float)

b = np.array([106.8,177.2,279.2], dtype=float)

x, det = gauss_elim(A, b)

print("Solution:")
print(x)

print("\nNumPy Solution:")
print(np.linalg.solve(A, b))

print("\nDeterminant:")
print(det)

print("\nNumPy Determinant:")
print(np.linalg.det(A))

print("\nResidual ||Ax-b||:")
print(np.linalg.norm(A @ x - b))