import numpy as np

def solve_gaussian_elimination(A, b, system_name="System"):
    """
    Solves Ax = b using Gaussian Elimination with Partial Pivoting.
    Handles three cases: Unique solution, No solution, Infinite solutions.
    Prints all row swaps, pivots, and the augmented matrix after elimination.
    """
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)]).astype(float)

    print(f"\n{'='*50}")
    print(f"{system_name}")
    print(f"{'='*50}")
    print("\nInitial Augmented Matrix:")
    print(Ab)
    print("\n--- Forward Elimination ---")

    for k in range(n):
        # Partial pivoting
        max_idx = k + np.argmax(np.abs(Ab[k:, k]))
        pivot_val = Ab[max_idx, k]

        print(f"\nStep {k+1}: Column {k+1}, Chosen Pivot = {pivot_val:.4f} at Row {max_idx+1}")

        if max_idx != k:
            print(f"  -> Swapping Row {k+1} and Row {max_idx+1}")
            Ab[[k, max_idx]] = Ab[[max_idx, k]]

        if np.abs(Ab[k, k]) < 1e-12:
            print("  -> Pivot is effectively zero. Skipping elimination for this column.")
            continue

        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]

    print("\nAugmented Matrix after Forward Elimination:")
    print(np.round(Ab, 4))

    # --- Determine the type of solution ---
    print("\n--- Solution Analysis ---")
    for i in range(n):
        # Check if all coefficients in this row are zero
        all_coeff_zero = np.all(np.abs(Ab[i, :n]) < 1e-12)
        if all_coeff_zero:
            if np.abs(Ab[i, -1]) > 1e-12:
                # 0x + 0y + 0z = c (c != 0) => NO SOLUTION
                eq_str = " + ".join([f"0*x{j+1}" for j in range(n)])
                print(f"\nRow {i+1} resulted in: {eq_str} = {Ab[i, -1]:.4f}")
                print("This is IMPOSSIBLE (0 = non-zero).")
                print("\nResult: NO SOLUTION (Inconsistent System)")
                return
            else:
                # 0x + 0y + 0z = 0 => dependent row, infinite solutions
                pass

    # Check rank
    rank = 0
    for i in range(n):
        if not np.all(np.abs(Ab[i, :n]) < 1e-12):
            rank += 1

    if rank < n:
        # INFINITE SOLUTIONS
        print(f"\nRank of coefficient matrix = {rank} < {n} (number of unknowns)")
        print("Result: INFINITE SOLUTIONS (Dependent System)")
        print("\nChoosing free variable x3 = 5:")

        x = np.zeros(n)
        x[n - 1] = 5.0  # Free variable

        for i in range(rank - 1, -1, -1):
            # Find the pivot column for this row
            pivot_col = -1
            for j in range(n):
                if np.abs(Ab[i, j]) > 1e-12:
                    pivot_col = j
                    break
            if pivot_col != -1:
                x[pivot_col] = (Ab[i, -1] - np.dot(Ab[i, pivot_col+1:n], x[pivot_col+1:])) / Ab[i, pivot_col]

        for i in range(n):
            print(f"  x{i+1} = {x[i]:.4f}")
        return

    # UNIQUE SOLUTION — Back Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]

    print("\nResult: UNIQUE SOLUTION")
    for i in range(n):
        print(f"  x{i+1} = {x[i]:.4f}")

    # Verification with NumPy
    np_x = np.linalg.solve(A, b)
    print(f"\nNumPy Verification: {np.round(np_x, 4)}")

    residual = np.dot(A, x) - b
    print(f"||Ax - b||_2 = {np.linalg.norm(residual):.6e}")


if __name__ == "__main__":
    # System 1: Unique Solution
    A1 = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b1 = np.array([8, -11, -3], dtype=float)
    solve_gaussian_elimination(A1, b1, "SYSTEM 1: Unique Solution Case")

    # System 2: No Solution
    A2 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float)
    b2 = np.array([1, 2, 5], dtype=float)
    solve_gaussian_elimination(A2, b2, "SYSTEM 2: No Solution Case")

    # System 3: Infinite Solutions
    A3 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float)
    b3 = np.array([1, 2, 3], dtype=float)
    solve_gaussian_elimination(A3, b3, "SYSTEM 3: Infinite Solutions Case")
