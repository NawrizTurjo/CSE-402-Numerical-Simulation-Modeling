"""
Complex case: two REAL eigenvalues tied in magnitude, opposite sign
========================================================================
Classic textbook trap (Chapra & Canale): lambda1 = -lambda2 (e.g. +4, -4).
Both are equally "dominant" by magnitude, so the power method's core
assumption (ONE strictly-largest eigenvalue) fails. Symptom: the estimate
settles into a period-2 oscillation instead of converging -- and, unlike
the tidy special case where the oscillation lands exactly on +lambda/-lambda,
in general the two oscillating VALUES are neither one individually (they're
a mix determined by how x0 projects onto the two tied eigenvectors).

Fix: run the power method on A^2 instead of A. Since lambda1=-lambda2,
lambda1^2 = lambda2^2 -- the tie becomes a REPEATED (but now unique-
magnitude, all-positive) dominant eigenvalue of A^2, which converges
cleanly. Take sqrt() of the result to recover |lambda|. (Sign of the
original lambda is then a separate question -- see the note at the bottom.)
"""
import numpy as np


def power_iteration(A, x0, max_iter=16, verbose=True):
    x = x0.astype(float).copy()
    history = []
    for it in range(max_iter):
        y = A @ x
        lam_est = y[np.argmax(np.abs(y))]
        history.append(lam_est)
        x = y / lam_est
        if verbose:
            print(f"  iter {it+1:2d}: eig est = {lam_est: .5f}   x = {np.round(x, 4)}")
    return history


if __name__ == "__main__":
    # constructed so eigenvalues are exactly +4, -4 (non-permutation
    # eigenvectors, so the oscillation isn't trivially clean)
    V = np.array([[1, 2], [3, 1]], dtype=float)
    Lam = np.diag([4.0, -4.0])
    A = V @ Lam @ np.linalg.inv(V)
    print("A =\n", A)
    print("true eigenvalues:", np.linalg.eig(A)[0])

    print("\n" + "=" * 60, "\nPlain power method -- oscillates, never converges\n", "=" * 60)
    x0 = np.array([1, 0.3], dtype=float)
    power_iteration(A, x0)
    print("\n^ settles into a 2-cycle -- NEITHER value is exactly +4 or -4;")
    print("  they're a mixture. This is the tell that you have a tied pair.")

    print("\n", "=" * 60, "\nFix: power method on A^2 -- converges cleanly\n", "=" * 60)
    history2 = power_iteration(A @ A, x0, max_iter=8)
    magnitude = np.sqrt(history2[-1])
    print(f"\nA^2's dominant eigenvalue -> {history2[-1]:.4f}  =>  |lambda| = sqrt(...) = {magnitude:.4f}")

    print("\n", "=" * 60, "\nRecovering the sign / actual eigenvectors\n", "=" * 60)
    print("""
The A^2 trick only recovers the MAGNITUDE (both +lambda and -lambda square
to the same thing, so A^2 can't distinguish them). To get the actual signed
eigenvalues and their eigenvectors:
  1. Solve (A - lambda*I)v = 0 and (A + lambda*I)v = 0 directly (null space),
     using the magnitude found above, or
  2. Just fall back to np.linalg.eig(A) -- once you've DIAGNOSED a tie
     (oscillating power method), reporting "power method finds |lambda|=4,
     tied between +4 and -4; np.linalg.eig confirms both" is a complete and
     correct exam answer.
""")
    print("np.linalg.eig confirms:", np.linalg.eig(A)[0])
