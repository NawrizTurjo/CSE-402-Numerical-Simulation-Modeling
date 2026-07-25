"""
Complex case: ill-conditioned systems -- small residual, big error
=======================================================================
A small ||Ax-b|| does NOT guarantee x is close to the true answer. The
relevant quantity is the condition number kappa(A) = ||A|| * ||A^-1||.
For ill-conditioned A, tiny input perturbations (or rounding error) cause
large output changes, even though the residual looks fine.

Classic example: the Hilbert matrix, H[i,j] = 1/(i+j+1) -- famously
ill-conditioned, gets dramatically worse as n grows.
"""
import numpy as np


def hilbert_matrix(n):
    return np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])


def forward_elimination(A, b):
    A = A.astype(float).copy(); b = b.astype(float).copy(); n = len(b)
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(A[k:, k])))
        if p != k:
            A[[k, p]] = A[[p, k]]; b[k], b[p] = b[p], b[k]
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]; b[i] -= m * b[k]
    return A, b


def back_substitution(U, c):
    n = len(c); x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (c[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x


if __name__ == "__main__":
    for n in [3, 6, 10]:
        print("=" * 60, f"\nHilbert matrix, n={n}\n", "=" * 60)
        H = hilbert_matrix(n)
        x_true = np.ones(n)              # known exact answer
        b = H @ x_true                    # construct a consistent RHS

        U, c = forward_elimination(H, b)
        x = back_substitution(U, c)

        residual = np.linalg.norm(H @ x - b)
        error = np.linalg.norm(x - x_true)
        kappa = np.linalg.cond(H)

        print(f"condition number kappa(H)  = {kappa:.4e}")
        print(f"residual ||Hx-b||_2        = {residual:.4e}   (looks small/fine!)")
        print(f"actual error ||x-x_true||_2 = {error:.4e}   (grows with kappa!)")
        print(f"error/residual ratio        = {error / max(residual, 1e-300):.4e}  (~ bounded by kappa)")

    print("\n" + "=" * 60)
    print("TAKEAWAY: as n grows, kappa(H) explodes (Hilbert matrices are the")
    print("textbook example of ill-conditioning). A tiny residual at n=10 can")
    print("coexist with an x that's completely wrong. ALWAYS report kappa(A)")
    print("alongside a residual if a question asks you to 'verify' a solution.")
