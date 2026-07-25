def solve_gauss_partial_pivoting():
    """
    Solves a system of n linear equations Ax = b using Gauss Elimination 
    with Partial Pivoting.
    """
    print("--- Gauss Elimination with Partial Pivoting ---")
    try:
        n = int(input("Enter the number of equations (n): ").strip())
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return

    A = []
    print(f"Enter the {n}x{n} matrix A row by row (space-separated values):")
    for i in range(n):
        row = list(map(float, input(f"Row {i+1}: ").strip().split()))
        if len(row) != n:
            print(f"Error: Expected {n} values, got {len(row)}.")
            return
        A.append(row)
        
    b = []
    print(f"Enter the {n}x1 vector b (one value per line):")
    for i in range(n):
        val = float(input(f"b[{i+1}]: ").strip())
        b.append(val)

    print("\nStarting Elimination Process...")
    # Forward Elimination with Partial Pivoting
    for k in range(n - 1):
        # Search for the largest absolute value in column k, from row k to n-1
        max_idx = k
        max_val = abs(A[k][k])
        for i in range(k + 1, n):
            if abs(A[i][k]) > max_val:
                max_val = abs(A[i][k])
                max_idx = i
                
        # Swap rows if a larger pivot was found
        if max_idx != k:
            print(f"Swapping row {k+1} and row {max_idx+1} (pivot = {A[max_idx][k]})")
            A[k], A[max_idx] = A[max_idx], A[k]
            b[k], b[max_idx] = b[max_idx], b[k]
            
        # Check for division by zero (singular matrix)
        if abs(A[k][k]) < 1e-12:
            print("Error: Matrix is singular or nearly singular (zero pivot).")
            return
            
        # Eliminate entries below the pivot
        for i in range(k + 1, n):
            multiplier = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= multiplier * A[k][j]
            b[i] -= multiplier * b[k]
            
    # Back Substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = sum(A[i][j] * x[j] for j in range(i + 1, n))
        
        if abs(A[i][i]) < 1e-12:
            print("Error: Matrix is singular or nearly singular (zero diagonal).")
            return
            
        x[i] = (b[i] - sum_ax) / A[i][i]
        
    print("\nFinal Solution vector x:")
    for i in range(n):
        print(f"x[{i+1}] = {x[i]:.6f}")

if __name__ == "__main__":
    solve_gauss_partial_pivoting()
