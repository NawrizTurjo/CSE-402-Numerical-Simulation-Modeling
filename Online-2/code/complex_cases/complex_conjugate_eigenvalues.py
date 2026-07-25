"""
Complex case: dominant eigenvalue pair is complex (not real)
=================================================================
The power method's convergence proof assumes a single REAL dominant
eigenvalue strictly larger in magnitude than the rest. If the two largest-
magnitude eigenvalues are a complex-conjugate pair a+bi, a-bi (equal
magnitude, by definition of "conjugate"), that assumption breaks: there is
no single dominant direction to converge to. The vector direction ROTATES
every iteration instead of settling, and the eigenvalue estimate cycles
through a repeating pattern instead of converging.

This IS something that can appear in this syllabus: any non-symmetric
real matrix can have complex eigenvalues (symmetric matrices are
guaranteed real eigenvalues -- non-symmetric ones are not).
"""
import numpy as np


def power_iteration(A, x0, max_iter=12, verbose=True):
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


def is_oscillating(history, tail=8, tol=1e-4):
    if len(history) < tail + 1:
        return False
    recent = history[-tail:]
    sign_changes = sum(1 for i in range(len(recent) - 1) if recent[i] * recent[i + 1] < 0)
    not_converged = abs(recent[-1] - recent[-2]) > tol
    return sign_changes >= tail // 2 and not_converged


if __name__ == "__main__":
    print("=" * 60, "\nPure rotation matrix -- eigenvalues +-i (magnitude 1, period 4)\n", "=" * 60)
    A1 = np.array([[0, -1], [1, 0]], dtype=float)
    h1 = power_iteration(A1, np.array([1, 0], dtype=float))
    print("\nestimates never converge -- they cycle through +1, -1, +1, -1, ...")
    print("np.linalg.eig ground truth:", np.linalg.eig(A1)[0])
    print("is_oscillating() ->", is_oscillating(h1))

    print("\n", "=" * 60, "\nSpiral matrix -- eigenvalues 1+-i (magnitude sqrt(2), period 4)\n", "=" * 60)
    A2 = np.array([[1, -1], [1, 1]], dtype=float)
    h2 = power_iteration(A2, np.array([1, 0], dtype=float))
    print("\nestimates cycle through 1, 2, -1, 2, 1, 2, -1, 2, ... -- never settle")
    print("np.linalg.eig ground truth:", np.linalg.eig(A2)[0], " |lambda| =", abs(np.linalg.eig(A2)[0][0]))
    print("is_oscillating() ->", is_oscillating(h2))

    print("\n", "=" * 60, "\nWhat to do about it live in the exam\n", "=" * 60)
    print("""
1. Recognize the symptom: eigenvalue estimates settle into a repeating
   cycle (period > 1) instead of converging to a single number. Check with
   is_oscillating() or just eyeball the printed history.
2. This means A has a complex-conjugate dominant pair -- there is no real
   dominant eigenvector for plain power iteration to find.
3. If the question just wants "the dominant eigenvalue" and A turns out to
   have complex eigenvalues, say so explicitly and fall back to
   np.linalg.eig(A) -- report the complex pair directly (they ARE the
   correct answer; "no real dominant eigenvalue exists" is itself a valid
   and often-expected written-question answer).
4. Symmetric matrices are IMMUNE to this -- symmetric A always has real
   eigenvalues. If your matrix is symmetric and you're still seeing
   oscillation, the bug is almost certainly in your code (check the
   argmax(abs(.)) pattern), not the math.
""")
