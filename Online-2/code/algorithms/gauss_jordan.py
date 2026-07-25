"""
Gauss-Jordan Elimination -- Canonical Library Functions & Theoretical Mappings
==============================================================================

THEORETICAL ALGORITHM MAPPING:
------------------------------
Transforms Augmented Matrix [A | b] directly into Reduced Row Echelon Form (RREF) [I | x].

1. AUGMENTATION:
   Aug = [A | b] in R^(n x (n+1))

2. FOR EACH COLUMN k = 0, 1, ..., n-1:
   a) Partial Pivoting Strategy:
      Locate pivot row p >= k such that |Aug[p, k]| is maximized.
      Swap Row k <-> Row p.

   b) Pivot Normalization (Scale pivot row so pivot element becomes 1.0):
      Row_k <- Row_k / Aug[k, k]
      Aug[k, j] = Aug[k, j] / Aug[k, k]   for j = k ... n

   c) Elimination Above & Below Pivot:
      For EVERY row i = 0, 1, ..., n-1 (where i != k):
        Row_i <- Row_i - Aug[i, k] * Row_k
        Aug[i, j] = Aug[i, j] - Aug[i, k] * Aug[k, j]   for j = k ... n

3. OUTPUT:
   The augmented matrix reaches [I_n | x], where x is the exact solution vector.
   No back substitution step is required!
"""
import numpy as np

EPS = 1e-9


def gauss_jordan(A, b, partial_pivot=True, hand_written=True, verbose=False):
    """
    Solves A * x = b by driving [A | b] to [I | x] using Gauss-Jordan elimination.

    THEORETICAL STEPS:
    1. Construct augmented matrix Aug = [A | b].
    2. Loop over each pivot column k = 0 to n-1.
    3. Perform partial pivoting row swap if enabled.
    4. Normalize pivot row k by dividing by pivot element Aug[k, k].
    5. Eliminate all non-zero entries in column k for both row i < k and row i > k.

    Returns:
        x (ndarray): Solution vector of shape (n,).
        Aug (ndarray): Final transformed augmented matrix [I | x].
    """
    n = len(b)
    # Construct initial augmented matrix [A | b] of shape (n, n+1)
    Aug = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for k in range(n):
        # --- PARTIAL PIVOTING: Find p >= k maximizing |Aug[p, k]| ---
        if partial_pivot:
            if hand_written:
                p = k
                for i in range(k + 1, n):
                    if abs(Aug[i, k]) > abs(Aug[p, k]):
                        p = i
            else:
                p = k + int(np.argmax(np.abs(Aug[k:, k])))

            if p != k:
                if verbose:
                    print(f"  row swap: R{k} <-> R{p}")
                if hand_written:
                    for j in range(n + 1):
                        Aug[k, j], Aug[p, j] = Aug[p, j], Aug[k, j]
                else:
                    Aug[[k, p]] = Aug[[p, k]]

        # --- SINGULARITY CHECK: Pivot cannot be zero ---
        if abs(Aug[k, k]) < EPS:
            raise ZeroDivisionError(f"pivot at column {k} is ~0 -- matrix is singular")

        # --- PIVOT NORMALIZATION: Row_k <- Row_k / Aug[k, k] ---
        # Theoretical goal: Make entry Aug[k, k] == 1.0
        if hand_written:
            pivot = Aug[k, k]
            for j in range(k, n + 1):
                Aug[k, j] = Aug[k, j] / pivot
        else:
            Aug[k] = Aug[k] / Aug[k, k] # <- Normalizing the entire row

        # --- ELIMINATION ABOVE AND BELOW PIVOT ---
        # Theoretical goal: Eliminate entry in column k for ALL rows i != k
        for i in range(n):
            if i != k:
                if hand_written:
                    factor = Aug[i, k]
                    for j in range(k, n + 1):
                        Aug[i, j] -= factor * Aug[k, j]
                else:
                    # Formula: Row_i <- Row_i - Aug[i, k] * Row_k
                    Aug[i] -= Aug[i, k] * Aug[k] # <- This is how we eliminate the column

        if verbose:
            print(f"after clearing column {k}:\n{np.round(Aug, 4)}")

    # Extract final column as solution vector x = Aug[:, -1]
    return Aug[:, -1], Aug


def _self_test():
    """Self-test using slides worked example (x = [3, -2.5, 7])."""
    A = np.array([[3, -0.1, -0.2], [0.1, 7, -0.3], [0.3, -0.2, 10]], dtype=float)
    b = np.array([7.85, -19.3, 71.4], dtype=float)
    x, _ = gauss_jordan(A, b)
    assert np.allclose(x, [3, -2.5, 7], atol=1e-4), x
    print("gauss_jordan.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()