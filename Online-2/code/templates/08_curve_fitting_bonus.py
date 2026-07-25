"""
Bonus: building a linear system from data points (curve fitting)
=====================================================================
Seen in past exams as a wrapper around Gauss/LU: fit v(t) = a1*t^2 + a2*t + a3
through 3 (t, v) points -- 3 equations, 3 unknowns, solve exactly like any
other Ax=b. Generalizes to any polynomial degree / any basis functions.

Pattern: np.column_stack([...]) builds the coefficient matrix one column
per basis function, evaluated at every data point.
"""
import numpy as np


def design_matrix_polynomial(t, degree):
    """[t^degree, ..., t^2, t, 1] as columns -- highest power first."""
    cols = [t ** p for p in range(degree, -1, -1)]
    return np.column_stack(cols)


def fit_polynomial_gauss(t, y, degree):
    """Solves for polynomial coefficients via naive Gauss elimination
    (no pivoting -- fine as long as the design matrix isn't singular)."""
    A = design_matrix_polynomial(np.asarray(t, dtype=float), degree)
    b = np.asarray(y, dtype=float)
    n = len(b)
    A = A.copy()
    b = b.copy()

    for k in range(n - 1):
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x   # coefficients, highest power first


if __name__ == "__main__":
    # rocket velocity example from the slides: v(t) = a1*t^2 + a2*t + a3
    t = [5, 8, 12]
    v = [106.8, 177.2, 279.2]

    coeffs = fit_polynomial_gauss(t, v, degree=2)
    print("coefficients [a1, a2, a3] =", np.round(coeffs, 6))

    A_check = design_matrix_polynomial(np.array(t, dtype=float), 2)
    print("numpy check:", np.round(np.linalg.solve(A_check, v), 6))

    # evaluate the fitted curve anywhere
    def evaluate(coeffs, t_new):
        degree = len(coeffs) - 1
        powers = np.array([t_new ** p for p in range(degree, -1, -1)])
        return coeffs @ powers

    for t_test in [6, 7.5, 9, 11]:
        print(f"v({t_test}) = {evaluate(coeffs, t_test):.4f}")
