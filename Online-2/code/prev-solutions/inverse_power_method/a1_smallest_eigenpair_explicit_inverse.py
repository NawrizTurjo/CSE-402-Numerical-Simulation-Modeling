"""
A1's actual online-exam question (transcribed in res/Online Questions.txt):

    A = [[4, 1, 0, 0],
         [1, 3, 1, 0],
         [0, 1, 2, 1],
         [0, 0, 1, 1]]

    "Find the smallest eigenvalue and eigenvector using the inverse power
    method, using the library function np.linalg.inv(). Compute the
    eigenvalues and eigenvectors using NumPy. Compare the results."

None of the collected friend-solutions used this exact matrix with an
explicit np.linalg.inv() call (see code/prev-solutions/README.md for what
was actually collected vs. what was really asked), so this file solves the
real question directly, built on algorithms/inverse_power_method.py's
`inverse_power_naive`.
"""
import numpy as np


def inverse_power_naive(A, x0, tol=1e-8, max_iter=1000, verbose=True):
    Ainv = np.linalg.inv(A)          # question explicitly names this function
    x = x0.astype(float).copy()
    history = []
    for it in range(max_iter):
        y = Ainv @ x
        lam_est = y[np.argmax(np.abs(y))]
        history.append(lam_est)
        x_new = y / lam_est
        if verbose:
            print(f"  iter {it+1:3d}: eig(A^-1) est = {lam_est:10.6f}  ->  eig(A) est = {1/lam_est:10.6f}")
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new
    x = x / np.linalg.norm(x)
    return 1.0 / history[-1], x, history


if __name__ == "__main__":
    A = np.array([[4, 1, 0, 0],
                  [1, 3, 1, 0],
                  [0, 1, 2, 1],
                  [0, 0, 1, 1]], dtype=float)
    x0 = np.array([1, 1, 1, 1], dtype=float)   # question doesn't specify x0; [1,1,1,1] works fine here

    print("A =\n", A)
    print("\n=== Inverse power method (explicit np.linalg.inv, as asked) ===")
    lam_min, v, history = inverse_power_naive(A, x0)
    print(f"\nsmallest eigenvalue : {lam_min:.6f}")
    print("eigenvector         :", np.round(v, 6))
    print("iterations          :", len(history))

    print("\n=== NumPy verification (as the question asks) ===")
    eigvals, eigvecs = np.linalg.eig(A)
    k = np.argmin(np.abs(eigvals))
    lam_np, v_np = eigvals[k].real, eigvecs[:, k].real
    print(f"smallest eigenvalue : {lam_np:.6f}")
    print("eigenvector         :", np.round(v_np, 6))

    if np.dot(v, v_np) < 0:
        v_np = -v_np
        print("(sign flipped for comparison -- v and -v are equally valid eigenvectors)")

    print(f"\n|lambda difference|        : {abs(lam_min - lam_np):.2e}")
    print(f"||eigenvector difference||_2 : {np.linalg.norm(v - v_np):.2e}")
    print(f"residual ||Av - lam*v||_2   : {np.linalg.norm(A @ v - lam_min * v):.2e}")

    print("\n=== All eigenvalues, for context ===")
    print("np.linalg.eig(A) full spectrum:", np.round(np.sort(eigvals.real), 6))
