import numpy as np

def power_method(A, tol=1e-12):
    A = A.astype(float)
    n = len(A)

    x = np.ones(n)

    prev = None

    while True:

        y = A @ x

        eig = np.max(np.abs(y))

        x = y / eig

        if prev is not None:
            err = abs((eig - prev) / eig)
            if err < tol:
                break

        prev = eig

    return eig, x


def deflation(A, eig, v):

    v = v / np.linalg.norm(v)

    return A - eig * np.outer(v, v)


A = np.array([[4,2],
              [2,3]], dtype=float)

# Largest eigenpair
eig1, v1 = power_method(A)

# Second eigenpair
A2 = deflation(A, eig1, v1)

eig2, v2 = power_method(A2)

# Normalize eigenvectors
v1 = v1 / np.linalg.norm(v1)
v2 = v2 / np.linalg.norm(v2)

# Construct V and Lambda
V = np.column_stack((v1, v2))

Lambda = np.diag([eig1, eig2])

V_inv = np.linalg.inv(V)

# Verify A = VΛV⁻¹
A_check = V @ Lambda @ V_inv

print("Original A")
print(A)

print("\nReconstructed A")
print(A_check)

print("\nVerification Error")
print(np.linalg.norm(A - A_check))

# Compute A^5
k = 5

Lambda_k = np.diag(np.power([eig1, eig2], k))

A_power = V @ Lambda_k @ V_inv

print("\nA^5 using Eigen Decomposition")
print(A_power)

print("\nA^5 using NumPy")
print(np.linalg.matrix_power(A, k))

print("\nError")
print(np.linalg.norm(A_power - np.linalg.matrix_power(A, k)))


# eigvals, V = np.linalg.eig(A)

# Lambda = np.diag(eigvals)

# A_check = V @ Lambda @ np.linalg.inv(V)