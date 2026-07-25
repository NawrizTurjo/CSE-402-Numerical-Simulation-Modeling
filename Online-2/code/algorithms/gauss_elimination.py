"""
Gauss Elimination -- Canonical Library Functions & Theoretical Mappings
========================================================================

THEORETICAL ALGORITHM MAPPING:
------------------------------
Solving Linear System: A * x = b, where A in R^(n x n) and b in R^n.

1. FORWARD ELIMINATION (Transforming [A | b] -> [U | c]):
   For step k = 0, 1, ..., n-2:
   - Partial Pivoting Strategy:
     Locate pivot index p in {k, ..., n-1} such that:
       |a_{p,k}^(k)| = max_{i >= k} |a_{i,k}^(k)|
     Swap Row k and Row p in both matrix A and vector b.
     *Why?* Prevents division by zero when a_{k,k}=0 and minimizes floating-point
     roundoff amplification by ensuring multiplier magnitude |m_{i,k}| <= 1.

   - Multiplier Computation:
       m_{i,k} = a_{i,k}^(k) / a_{k,k}^(k)   for i = k+1, ..., n-1

   - Row Operation Formula (Elimination):
       Row_i <- Row_i - m_{i,k} * Row_k
       a_{i,j}^(k+1) = a_{i,j}^(k) - m_{i,k} * a_{k,j}^(k)   for j = k, ..., n-1
       b_i^(k+1)     = b_i^(k)     - m_{i,k} * b_k^(k)

2. BACK SUBSTITUTION (Solving Upper Triangular System U * x = c):
   Starting from i = n-1 down to 0:
       x_n = c_n / u_{n,n}
       x_i = (c_i - sum_{j=i+1}^{n-1} u_{i,j} * x_j) / u_{i,i}

3. SYSTEM CLASSIFICATION (Rouche-Capelli Theorem):
   - Unique Solution:   rank(A) == rank([A|b]) == n (no zero rows on diagonal)
   - Inconsistent (No Sol): rank(A) < rank([A|b]) -> Row [0 0 ... 0 | c] with c != 0
   - Infinite Solutions: rank(A) == rank([A|b]) < n -> Zero row [0 ... 0 | 0], free variables exist.

4. DETERMINANT COMPUTATION VIA GAUSS ELIMINATION:
   - Row Swaps: Each row swap multiplies det(A) by -1.
   - Elimination: Adding a scalar multiple of one row to another preserves det(A).
   - Upper Triangular: det(U) = product_{i=0}^{n-1} u_{i,i}.
   - Formula: det(A) = (-1)^(number_of_swaps) * prod(diag(U)).
"""
import numpy as np

EPS = 1e-9 # <- to handle division by zero error


def forward_elimination(A, b, hand_written=True, verbose=False):
    """
    Gauss forward elimination with partial pivoting.

    THEORETICAL STEPS:
    1. Copy matrix A and vector b to avoid mutating caller data.
    2. Iterate through columns k = 0 to n-2 as pivot columns.
    3. Perform Partial Pivoting (find max magnitude in column k below diagonal).
    4. Compute elimination multiplier m = A[i, k] / A[k, k].
    5. Update row i: R_i <- R_i - m * R_k.

    Returns:
        U (ndarray): Upper triangular matrix (n x n).
        c (ndarray): Transformed right-hand side vector (n,).
    """
    # Create float copies of input arrays to preserve original inputs
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    n = len(b)

    # Step k: Eliminate entries under diagonal in column k (k = 0 ... n-2)
    for k in range(n - 1):
        
        # --- PARTIAL PIVOTING: Find p >= k maximizing |A[p, k]| ---
        if hand_written:
            # Theoretical Loop: Explicit manual search for largest pivot magnitude
            p = k
            for i in range(k + 1, n):
                if abs(A[i, k]) > abs(A[p, k]):
                    p = i # <- gets the maximum row's index in p (otherwise p=k, not updated) 
        else:
            # Vectorized NumPy equivalent using argmax over sub-array A[k:, k]
            p = k + int(np.argmax(np.abs(A[k:, k]))) # <- same as above but vectorized. O(n)

        if verbose:
            print(f"  column {k}: pivot = {A[p, k]:.6g} (row {p})")

        # --- ROW SWAP OPERATION: Swap Row k with Row p if p != k ---
        if p != k:
            if verbose:
                print(f"  row swap: R{k} <-> R{p}")
            if hand_written:
                # Theoretical Loop: Manual element-by-element row swapping
                for j in range(n):
                    A[k, j], A[p, j] = A[p, j], A[k, j]
            else:
                # Vectorized NumPy fancy indexing row swap
                A[[k, p]] = A[[p, k]]
            
            # Swap corresponding right-hand side elements in vector b
            b[k], b[p] = b[p], b[k]

        # --- SINGULARITY / ZERO PIVOT CHECK ---
        if abs(A[k, k]) < EPS:
            if verbose:
                print(f"  pivot in column {k} is 0 -> skip elimination for this column")
            continue # <- avoids division by zero error

        # --- ROW ELIMINATION LOOP: For each row i below pivot row k ---
        for i in range(k + 1, n):
            # Formula: multiplier m_{i,k} = a_{i,k} / a_{k,k}
            m = A[i, k] / A[k, k]

            # Formula: Row_i <- Row_i - m_{i,k} * Row_k  for columns j = k ... n-1
            A[i, k:] -= m * A[k, k:] # <- k: used to reduce time (ig)

            # Formula: b_i <- b_i - m_{i,k} * b_k
            b[i] -= m * b[k]

    return A, b


def back_substitution(U, c, hand_written=True):
    """
    Back substitution algorithm for upper triangular system U * x = c.

    THEORETICAL FORMULA:
    x_i = (c_i - sum_{j=i+1}^{n-1} U[i, j] * x_j) / U[i, i]
    iterating backwards from i = n-1 down to 0.

    Returns:
        x (ndarray): Solution vector of size n.
    """
    n = len(c)
    x = np.zeros(n)

    # Outer loop: Solve for unknowns backwards from x_{n-1} down to x_0
    for i in range(n - 1, -1, -1):
        if hand_written:
            total_sum = 0.0
            for j in range(i + 1, n):
                total_sum += U[i, j] * x[j]  # Calculates sum(u_ij * x_j)

            x[i] = (c[i] - total_sum) / U[i, i]  # Final division
        else:
            # Dot product U[i, i+1:] @ x[i+1:] evaluates sum_{j=i+1}^{n-1} U[i, j] * x_j
            x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

    

    return x


def classify_and_solve(A, b, free_value=1.0, hand_written=True, verbose=False):
    """
    General classifier & solver: handles unique, inconsistent, and rank-deficient systems.

    THEORETICAL CLASSIFICATION CRITERIA (Rouche-Capelli Theorem):
    1. Forward elimination reduces [A | b] to upper triangular [U | c].
    2. Examine each row i from bottom (i = n-1) to top (i = 0):
       - If row U[i] is ALL ZEROS (|U[i, j]| < EPS for all j):
         a) If |c[i]| > EPS: Equation is 0*x_0 + ... + 0*x_{n-1} = c_i (c_i != 0).
            System is INCONSISTENT -> "no_solution".
         b) If |c[i]| <= EPS: Equation is 0*x_0 + ... + 0*x_{n-1} = 0.
            Variable x_i is a FREE VARIABLE. Assign arbitrary free_value (e.g. 1.0).
       - Else (|U[i, i]| >= EPS):
         Calculate x_i normally using back substitution formula.

    Returns:
        Tuple (status, solution_vector, index_list):
        - ("unique", x, [])
        - ("no_solution", None, [inconsistent_row_indices])
        - ("infinite", x, [free_variable_indices])
    """
    # Step 1: Forward elimination to get upper triangular matrix U and RHS c
    U, c = forward_elimination(A, b, hand_written=hand_written, verbose=verbose)
    n = len(c)
    x = np.zeros(n)
    free_idx = []
    inconsistent = []

    # Step 2: Backward pass examining rows for zero rows & solving
    for i in range(n - 1, -1, -1):
        # Check if entire row U[i] is zero (rank deficiency)
        if hand_written:
            row_is_zero = True
            for j in range(n):
                if abs(U[i, j]) >= EPS:
                    row_is_zero = False
                    break
        else:
            row_is_zero = np.all(np.abs(U[i]) < EPS)

        if row_is_zero:
            if abs(c[i]) > EPS:
                # Contradiction: 0 = non-zero constant -> No Solution
                inconsistent.append(i)
                continue
            # Identity: 0 = 0 -> Infinite solutions, x_i is free variable
            x[i] = free_value
            free_idx.append(i)
        else:
            # Standard pivot non-zero: Solve for x_i using back substitution
            if hand_written:
                total_sum = 0.0
                for j in range(i + 1, n):
                    total_sum += U[i, j] * x[j]
                x[i] = (c[i] - total_sum) / U[i, i]
            else:
                x[i] = (c[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]

    # Return classification decision
    if inconsistent:
        return "no_solution", None, inconsistent
    if free_idx:
        return "infinite", x, free_idx
    return "unique", x, []


def determinant(A, hand_written=True, verbose=False):
    """
    Computes det(A) using Gaussian elimination with partial pivoting.

    THEORETICAL FORMULA:
    det(A) = (-1)^(number_of_swaps) * product_{i=0}^{n-1} U[i, i]

    Properties used:
    - Row swap R_k <-> R_p multiplies determinant by -1.
    - Row addition R_i <- R_i - m * R_k does NOT change determinant.
    - Determinant of triangular matrix U is the product of its diagonal elements.
    """
    A = A.astype(float).copy()
    n = A.shape[0]
    swaps = 0

    for k in range(n - 1):
        # Partial pivoting: Locate row p with largest pivot magnitude
        if hand_written:
            p = k
            for i in range(k + 1, n):
                if abs(A[i, k]) > abs(A[p, k]):
                    p = i
        else:
            p = k + int(np.argmax(np.abs(A[k:, k])))

        if p != k:
            if hand_written:
                for j in range(n):
                    A[k, j], A[p, j] = A[p, j], A[k, j]
            else:
                A[[k, p]] = A[[p, k]]
            swaps += 1  # Track row swap count for (-1)^swaps factor

        # Singular matrix check: If pivot is zero, det(A) = 0
        if abs(A[k, k]) < EPS:
            return 0.0

        # Elimination loop
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            if hand_written:
                for j in range(k, n):
                    A[i, j] -= m * A[k, j]
            else:
                A[i, k:] -= m * A[k, k:]

    # Formula: (-1)^swaps * prod(diag(U))
    det_val = 1
    for i in range (n):
        det_val *= A[i,i]

    # det = ((-1) ** swaps) * np.prod(np.diag(A))
    det = ((-1) ** swaps) * det_val
    if verbose:
        print(f"swaps={swaps}, diag(U)={np.round(np.diag(A), 4)}, det={det:.6g}")
    return det


def _self_test():
    """Numerical self-tests to verify implementation correctness."""
    # Test 1: System with a Unique Solution
    A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    kind, x, _ = classify_and_solve(A, b)
    assert kind == "unique" and np.allclose(x, [2, 3, -1]), x

    # Test 2: Inconsistent System (No Solution)
    A2 = np.array([[1, 1, 1], [2, 2, 2], [1, 2, 3]], dtype=float)
    b2 = np.array([1, 3, 4], dtype=float)
    kind, x, rows = classify_and_solve(A2, b2)
    assert kind == "no_solution"

    # Test 3: Infinite Solutions System (Rank-deficient by 2)
    A3 = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [1, -1, 1, -1], [2, -2, 2, -2]], dtype=float)
    b3 = np.array([4, 8, 0, 0], dtype=float)
    kind, x, free = classify_and_solve(A3, b3)
    assert kind == "infinite" and len(free) == 2 and np.allclose(A3 @ x, b3)

    # Test 4: Determinant Verification vs NumPy
    A4 = np.array([[0, 2, 1], [1, -2, -3], [-1, 1, 2]], dtype=float)
    assert abs(determinant(A4) - np.linalg.det(A4)) < 1e-8

    print("gauss_elimination.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()
