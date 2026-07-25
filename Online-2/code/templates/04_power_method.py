"""
Power Method -- dominant eigenvalue & eigenvector
====================================================
Steps (from slides):
  1. y_{k+1} = A x_k
  2. eigenvalue estimate = entry of LARGEST MAGNITUDE in y_{k+1} (keep sign!)
  3. x_{k+1} = y_{k+1} / (that entry)   -- normalizing convention
  4. repeat until consecutive eigenvalue estimates differ by < tol

Gotchas:
  - Use np.argmax(np.abs(y)) then index back for the sign, NOT np.max(y)
    (np.max picks the largest SIGNED value, silently wrong for e.g. [-5, 2]).
  - Before returning/comparing to NumPy, normalize the eigenvector to unit
    length (np.linalg.norm) -- the power-iteration convention (largest entry
    = 1) is different from NumPy's convention (||v||_2 = 1).
  - NumPy eigenvector may have opposite sign; both v and -v are valid.
"""
import numpy as np


def power_iteration(A, x0, tol=1e-8, max_iter=1000, verbose=True):
    """Returns (dominant_eigenvalue, unit_eigenvector, history_of_estimates)."""
    x = x0.astype(float).copy()
    history = []

    for it in range(max_iter):
        y = A @ x
        lam_est = y[np.argmax(np.abs(y))]     # magnitude-based, sign preserved
        history.append(lam_est)
        x_new = y / lam_est                    # normalize: largest entry -> 1

        if verbose:
            print(f"  iter {it + 1:3d}: eig est = {lam_est:12.6f}   x = {np.round(x_new, 5)}")

        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new

    x = x / np.linalg.norm(x)   # unit length, comparable to np.linalg.eig
    return history[-1], x, history


if __name__ == "__main__":
    A = np.array([[8, 2, 0, 0],
                  [2, 8, 0, 0],
                  [0, 0, 3, 1],
                  [0, 0, 1, 3]], dtype=float)
    x0 = np.array([1, 1, 1, 1], dtype=float)

    print("=== our power iteration ===")
    lam, v, history = power_iteration(A, x0)
    print("\ndominant eigenvalue  :", lam)
    print("dominant eigenvector :", v)
    print("iterations           :", len(history))

    print("\n=== NumPy verification ===")
    eigvals, eigvecs = np.linalg.eig(A)
    k = np.argmax(np.abs(eigvals))
    lam_np, v_np = eigvals[k], eigvecs[:, k]
    print("dominant eigenvalue  :", lam_np)
    print("dominant eigenvector :", v_np)

    if np.dot(v, v_np) < 0:
        v_np = -v_np
        print("(sign flipped for comparison -- v and -v are equally valid)")

    print("\n|lambda diff| :", abs(lam - lam_np))
    print("||v diff||_2  :", np.linalg.norm(v - v_np))
    print("||Av - lam*v||_2 (residual) :", np.linalg.norm(A @ v - lam * v))

    # --- optional convergence plot (see plotting_template.py for more) ---
    try:
        import matplotlib.pyplot as plt
        plt.plot(range(1, len(history) + 1), history, marker='o')
        plt.xlabel("iteration"); plt.ylabel("eigenvalue estimate")
        plt.title("Power method convergence"); plt.grid(True)
        plt.savefig("power_method_convergence.png", dpi=120)
        print("\nsaved plot -> power_method_convergence.png")
    except ImportError:
        pass


# ----------------------------------------------------------------------
# Written-question quick answers
# ----------------------------------------------------------------------
# Q: Why does the normalizing factor converge to the dominant eigenvalue?
# A: Any x0 = c1*v1 + c2*v2 + ... . A^k x0 = c1*lam1^k*v1 + c2*lam2^k*v2+...
#    Factor out lam1^k: = lam1^k [c1*v1 + c2*(lam2/lam1)^k*v2 + ...]. Since
#    |lam1|>|lam_i|, each ratio (lam_i/lam1)^k -> 0, so A^k x0 -> lam1^k*c1*v1.
#    The vector direction converges to v1; the normalizing factor -> lam1.
# Q: What if the dominant eigenvalue is not unique / x0 has zero component
#    along v1?
# A: Convergence fails or is much slower. In practice a "generic" x0 (not
#    aligned with any symmetry of A) avoids the zero-component blind spot.
