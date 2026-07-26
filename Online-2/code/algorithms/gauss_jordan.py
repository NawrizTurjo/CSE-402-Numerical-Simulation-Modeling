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

4. MATRIX INVERSE -- THE SAME LOOP WITH A WIDER RIGHT-HAND BLOCK:
   Augment with the whole identity instead of a single vector:
     [A | I]  --(exactly the same column sweep)-->  [I | A^(-1)]
   Why it works: driving [A | I] to RREF applies one sequence of elementary row
   operations E_k ... E_2 E_1 with (E_k ... E_1) A = I, i.e. E_k ... E_1 = A^(-1).
   Those same operations hit the right-hand block, turning I into A^(-1).
   Equivalently: column j of the right block solves A x_j = e_j, so the n
   columns of the answer are the n solutions -- the identity's columns are just
   "n right-hand sides", exactly the C2 reuse idea with b1, b2 replaced by
   e_1 ... e_n.

5. DETERMINANT AS A BYPRODUCT -- USE THE *RAW* PIVOTS:
     det(A) = (-1)^(number_of_swaps) * prod_k (raw pivot at column k)
   "Raw" = the pivot value BEFORE step 2b divides the row by it. Normalizing a
   row by p divides the determinant by p, so by the time [A|b] reaches full
   RREF every pivot has been scaled to 1 and the determinant information is
   gone (prod(diag) == 1 always). Capture each pivot at the moment it is found.
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


def gauss_jordan_inverse(A, hand_written=True, verbose=False):
    """
    Computes A^(-1) by driving the augmented block [A | I] to [I | A^(-1)].

    THEORETICAL STEPS (identical sweep to gauss_jordan, just a wider RHS):
    1. Build Aug = [A | I] of shape (n, 2n).
    2. For each column k = 0 ... n-1:
       a) Partial pivoting: swap in the row with the largest |Aug[i, k]|.
       b) Normalize: Row_k <- Row_k / Aug[k, k]   (pivot becomes 1).
       c) Eliminate column k from EVERY other row i != k:
          Row_i <- Row_i - Aug[i, k] * Row_k
    3. The left block is now I; the right block Aug[:, n:] is A^(-1).

    A singular A is detected as a pivot that stays ~0 even after pivoting --
    there is genuinely nothing to swap in, so the inverse does not exist.

    Returns:
        A_inv (ndarray): The inverse, shape (n, n).
        Aug (ndarray): Final augmented matrix [I | A^(-1)], shape (n, 2n).
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    # --- AUGMENT WITH THE IDENTITY: [A | I], shape (n, 2n) ---
    if hand_written:
        Aug = np.zeros((n, 2 * n))
        for i in range(n):
            for j in range(n):
                Aug[i, j] = A[i, j]
            Aug[i, n + i] = 1.0  # <- the identity block, one 1 per row
    else:
        Aug = np.hstack([A.copy(), np.eye(n)])

    for k in range(n):
        # --- PARTIAL PIVOTING over the LEFT block only (column k of A) ---
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
                for j in range(2 * n):
                    Aug[k, j], Aug[p, j] = Aug[p, j], Aug[k, j]
            else:
                Aug[[k, p]] = Aug[[p, k]]

        # --- SINGULARITY CHECK: no usable pivot left in this column ---
        if abs(Aug[k, k]) < EPS:
            raise ZeroDivisionError(
                f"pivot at column {k} is ~0 even after pivoting -- "
                "A is singular, so A^-1 does not exist")

        # --- NORMALIZE the pivot row (must sweep all 2n columns) ---
        pivot = Aug[k, k]
        if hand_written:
            for j in range(2 * n):
                Aug[k, j] = Aug[k, j] / pivot
        else:
            Aug[k] = Aug[k] / pivot

        # --- ELIMINATE column k from every OTHER row (above and below) ---
        for i in range(n):
            if i != k:
                factor = Aug[i, k]
                if abs(factor) < EPS:
                    continue
                if hand_written:
                    for j in range(2 * n):
                        Aug[i, j] -= factor * Aug[k, j]
                else:
                    Aug[i] -= factor * Aug[k]

        if verbose:
            print(f"after clearing column {k}:\n{np.round(Aug, 4)}")

    # Left block is I; right block is the inverse
    return Aug[:, n:].copy(), Aug


def gauss_jordan_determinant(A, hand_written=True, verbose=False):
    """
    Computes det(A) as a byproduct of the Gauss-Jordan sweep.

    THEORETICAL FORMULA:
        det(A) = (-1)^(swaps) * prod_k (RAW pivot at column k)

    The pivot must be recorded BEFORE the row is normalized -- normalizing by
    p is a row-scaling operation, which divides the determinant by p. Once the
    matrix is in full RREF every pivot equals 1 and prod(diag) == 1 regardless
    of the true determinant, so the value is unrecoverable at that point.

    Returns:
        det (float): The determinant.
        raw_pivots (list): The pivot values used, in column order.
        swaps (int): Number of row swaps performed.
    """
    A = np.asarray(A, dtype=float).copy()
    n = A.shape[0]
    raw_pivots = []
    swaps = 0

    for k in range(n):
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
            swaps += 1

        pivot = A[k, k]
        # Zero pivot with nothing to swap in -> singular -> det == 0
        if abs(pivot) < EPS:
            if verbose:
                print(f"  column {k}: no nonzero pivot -> singular, det = 0")
            return 0.0, raw_pivots, swaps

        # RECORD THE RAW PIVOT NOW, before normalization destroys it
        raw_pivots.append(pivot)
        if verbose:
            print(f"  column {k}: raw pivot = {pivot:.6g}")

        # Normalize, then clear the column above and below (Gauss-Jordan)
        for j in range(n):
            A[k, j] = A[k, j] / pivot
        for i in range(n):
            if i != k and abs(A[i, k]) > EPS:
                factor = A[i, k]
                for j in range(n):
                    A[i, j] -= factor * A[k, j]

    det = (-1.0) ** swaps
    for pivot in raw_pivots:
        det *= pivot

    if verbose:
        print(f"  raw pivots = {[round(p, 6) for p in raw_pivots]}, "
              f"swaps = {swaps}, det = {det:.10g}")
    return det, raw_pivots, swaps


def gauss_jordan_rref(Aug, hand_written=True, verbose=False):
    """
    Reduces ANY matrix (square or not, singular or not) to Reduced Row Echelon
    Form. Use this instead of `gauss_jordan` when the system might be singular
    or rank-deficient -- `gauss_jordan` raises on a zero pivot, this one skips
    the column and moves on, which is what classification needs.

    THEORETICAL DIFFERENCE FROM THE SQUARE CASE:
    The pivot no longer has to sit on the diagonal. Track a separate `row`
    cursor: a column with no usable pivot is a FREE-VARIABLE column and the
    cursor stays put while the column pointer advances.

    Returns:
        R (ndarray): The RREF matrix.
        pivot_cols (list): Index of the pivot column for each pivot row.
        rank (int): Number of pivots == rank of the input matrix.
    """
    R = np.asarray(Aug, dtype=float).copy()
    rows, cols = R.shape
    pivot_cols = []
    row = 0

    for col in range(cols):
        if row >= rows:
            break

        # Find the largest |entry| in this column at or below the cursor
        if hand_written:
            p = row
            for i in range(row + 1, rows):
                if abs(R[i, col]) > abs(R[p, col]):
                    p = i
        else:
            p = row + int(np.argmax(np.abs(R[row:, col])))

        # No usable pivot here -> free column, cursor does NOT advance
        if abs(R[p, col]) < EPS:
            if verbose:
                print(f"  column {col}: no pivot (free variable column)")
            continue

        if p != row:
            if verbose:
                print(f"  row swap: R{row} <-> R{p}")
            for j in range(cols):
                R[row, j], R[p, j] = R[p, j], R[row, j]

        pivot = R[row, col]
        for j in range(cols):
            R[row, j] = R[row, j] / pivot

        for i in range(rows):
            if i != row and abs(R[i, col]) > EPS:
                factor = R[i, col]
                for j in range(cols):
                    R[i, j] -= factor * R[row, j]

        pivot_cols.append(col)
        if verbose:
            print(f"  column {col}: pivot row {row}, raw pivot = {pivot:.6g}")
            print(np.round(R, 6))
        row += 1

    return R, pivot_cols, len(pivot_cols)


def verify_gauss_jordan(A, b=None, x=None, A_inv=None, det_value=None, verbose=True):
    A = np.asarray(A, float)
    n = A.shape[0]
    ok = True

    if x is not None and b is not None:
        b_np, x_np = np.asarray(b, float), np.asarray(x, float)
        res = float(np.linalg.norm(A @ x_np - b_np))
        gap = float(np.linalg.norm(x_np - np.linalg.solve(A, b_np)))
        ok &= (res < 1e-8 and gap < 1e-8)

    if A_inv is not None:
        inv = np.asarray(A_inv, float)
        I_err = float(np.linalg.norm(A @ inv - np.eye(n)))
        gap = float(np.linalg.norm(inv - np.linalg.inv(A)))
        ok &= (I_err < 1e-8 and gap < 1e-8)

    if det_value is not None:
        lib_det = float(np.linalg.det(A))
        scale = max(1.0, abs(lib_det))
        ok &= (abs(det_value - lib_det) / scale < 1e-8)

    if verbose:
        print("\n--- Verify: Gauss-Jordan ---")
        print("Status: ", "PASS" if ok else "FAIL")

    return bool(ok)


def _self_test():
    """Self-test using slides worked example (x = [3, -2.5, 7])."""
    A = np.array([[3, -0.1, -0.2], [0.1, 7, -0.3], [0.3, -0.2, 10]], dtype=float)
    b = np.array([7.85, -19.3, 71.4], dtype=float)
    x, _ = gauss_jordan(A, b)
    assert np.allclose(x, [3, -2.5, 7], atol=1e-4), x
    assert verify_gauss_jordan(A, b=b, x=x, verbose=False)

    # Inverse via [A | I], on a matrix whose first pivot is 0 (needs a swap)
    B = np.array([[0, 1, 2], [1, -1, 3], [2, 4, 1]], dtype=float)
    B_inv, aug = gauss_jordan_inverse(B)
    assert np.allclose(B_inv, np.linalg.inv(B)), B_inv
    assert np.allclose(aug[:, :3], np.eye(3), atol=1e-12)   # left block is I
    assert np.allclose(B @ B_inv, np.eye(3), atol=1e-12)
    assert verify_gauss_jordan(B, A_inv=B_inv, verbose=False)

    # Determinant from raw pivots (B needs one swap -> sign flip)
    det, pivots, swaps = gauss_jordan_determinant(B)
    assert abs(det - np.linalg.det(B)) < 1e-8, (det, np.linalg.det(B))
    assert swaps == 1 and len(pivots) == 3
    assert verify_gauss_jordan(B, det_value=det, verbose=False)

    # Singular matrix -> det 0, and the inverse must refuse rather than return junk
    S = np.array([[1, 2, 3], [2, 4, 6], [1, 0, 1]], dtype=float)
    det_s, *_ = gauss_jordan_determinant(S)
    assert det_s == 0.0
    try:
        gauss_jordan_inverse(S)
        raise AssertionError("expected ZeroDivisionError on a singular matrix")
    except ZeroDivisionError:
        pass

    # RREF on a rank-deficient system: rank 2 < 3 unknowns -> one free column
    Aug = np.array([[1, 1, 1, 6], [2, 2, 2, 12], [1, -1, 1, 2]], dtype=float)
    R, pivot_cols, rank = gauss_jordan_rref(Aug)
    assert rank == 2 and pivot_cols == [0, 1], (rank, pivot_cols)
    assert np.allclose(R[2], 0.0, atol=1e-12)     # dependent row collapsed to 0 = 0

    print("gauss_jordan.py: all self-tests passed")


if __name__ == "__main__":
    _self_test()