import numpy as np

def solve_gaussian_elimination(A, b):
    n = len(b)
    # Augment matrix
    Ab = np.hstack([A, b.reshape(-1, 1)]).astype(float)
    
    print("\nInitial Augmented Matrix:")
    print(Ab)
    print("\n--- Forward Elimination ---")
    
    for k in range(n):
        # Partial pivoting: find max absolute value in the current column, from current row down
        max_idx = k + np.argmax(np.abs(Ab[k:, k]))
        pivot_val = Ab[max_idx, k]
        
        print(f"Step {k+1}: Column {k+1}, Chosen Pivot = {pivot_val} at Row {max_idx+1}")
        
        # Row swapping
        if max_idx != k:
            print(f"  -> Swapping Row {k+1} and Row {max_idx+1}")
            Ab[[k, max_idx]] = Ab[[max_idx, k]]
            
        if np.abs(Ab[k, k]) < 1e-12:
            print("  -> Pivot is effectively zero. Skipping elimination for this column.")
            continue
            
        # Eliminate entries below the pivot
        for i in range(k + 1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]
            
    print("\nAugmented Matrix after Forward Elimination:")
    print(np.round(Ab, 4))
    
    print("\n--- Checking Solutions ---")
    # Check from bottom to top for zero rows
    status = "Unique"
    for i in range(n):
        all_zeros = True
        for j in range(n):
            if np.abs(Ab[i, j]) > 1e-12:
                all_zeros = False
                break
        
        if all_zeros:
            if np.abs(Ab[i, -1]) > 1e-12:
                print(f"Row {i+1} resulted in an equation like 0.x + 0.y + 0.z = {Ab[i, -1]:.4f}")
                print("State: NO SOLUTION")
                return
            else:
                status = "Infinite"
    
    if status == "Infinite":
        print("Row(s) resulted in an equation like 0.x + 0.y + 0.z = 0")
        print("State: INFINITE SOLUTIONS")
        print("Choosing free variable x3 = 5 as per instructions.")
        
        x = np.zeros(n)
        x[2] = 5.0 # Hardcoded free variable
        
        # Back substitution for infinite solution (assumes rank is n-1, i.e., 2)
        for i in range(n - 2, -1, -1):
            if np.abs(Ab[i, i]) > 1e-12:
                sum_ax = sum(Ab[i, j] * x[j] for j in range(i + 1, n))
                x[i] = (Ab[i, -1] - sum_ax) / Ab[i, i]
        
        print(f"One possible solution with x3=5: x1={x[0]:.4f}, x2={x[1]:.4f}, x3={x[2]:.4f}")
        return

    # Unique Solution Back Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ax = sum(Ab[i, j] * x[j] for j in range(i + 1, n))
        x[i] = (Ab[i, -1] - sum_ax) / Ab[i, i]
        
    print("State: UNIQUE SOLUTION")
    print(f"Solution: {np.round(x, 4)}")
    
    # NumPy verification
    np_x = np.linalg.solve(A, b)
    print(f"NumPy Verification: {np.round(np_x, 4)}")
    
    # L2 Norm
    residual = np.dot(A, x) - b
    l2_norm = np.linalg.norm(residual)
    print(f"||Ax - b||_2 = {l2_norm:.6e}")


if __name__ == "__main__":
    print("=========================================")
    print("SYSTEM 1: UNIQUE SOLUTION CASE")
    A1 = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
    b1 = np.array([8, -11, -3], dtype=float)
    solve_gaussian_elimination(A1, b1)
    
    print("\n=========================================")
    print("SYSTEM 2: NO SOLUTION CASE")
    A2 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float)
    b2 = np.array([1, 2, 5], dtype=float)
    solve_gaussian_elimination(A2, b2)
    
    print("\n=========================================")
    print("SYSTEM 3: INFINITE SOLUTIONS CASE")
    A3 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float)
    b3 = np.array([1, 2, 3], dtype=float)
    solve_gaussian_elimination(A3, b3)
