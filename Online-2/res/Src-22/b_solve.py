import numpy as np

def gauss_elim(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()

    A_original = A.copy()
    b_original = b.copy()

    n = len(b)

    # ---------- Forward Elimination ----------
    for i in range(n):

        # Partial Pivoting
        max_row = i
        max_val = abs(A[i][i])

        for j in range(i + 1, n):
            if abs(A[j][i]) > max_val:
                max_val = abs(A[j][i])
                max_row = j

        print(f"\nPivot at column {i+1}: {A[max_row][i]}")

        if max_row != i:
            print(f"Swap R{i+1} <-> R{max_row+1}")

            A[[i, max_row]] = A[[max_row, i]]
            b[i], b[max_row] = b[max_row], b[i]

        for k in range(i + 1, n):

            if abs(A[i][i]) < 1e-12:
                continue

            factor = A[k][i] / A[i][i]

            for j in range(i, n):
                A[k][j] -= factor * A[i][j]

            b[k] -= factor * b[i]

    # ---------- Print Augmented Matrix ----------
    print("\nAugmented Matrix after Forward Elimination")

    for i in range(n):
        print(np.append(A[i], b[i]))

    # ---------- Check system ----------
    infinite = False

    for i in range(n):

        if np.all(np.abs(A[i]) < 1e-10):

            if abs(b[i]) > 1e-10:

                print("\nNo Solution")
                print("Contradictory equation:")

                eq = " + ".join([f"0.x{j+1}" for j in range(n)])
                print(eq + f" = {b[i]}")

                return

            else:
                infinite = True

    if infinite:

        print("\nInfinite Solutions")

        print("Choose free variable:")
        print("Let x3 = 5")

        x = np.zeros(n)
        x[2] = 5

        # Solve remaining equations
        for i in range(n - 2, -1, -1):

            rhs = b[i]

            for j in range(i + 1, n):
                rhs -= A[i][j] * x[j]

            x[i] = rhs / A[i][i]

        print("One possible solution:")
        print(x)

        return

    # ---------- Unique Solution ----------
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):

        x[i] = b[i]

        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]

        x[i] /= A[i][i]

    print("\nUnique Solution")
    print(x)

    print("\nVerification using np.linalg.solve()")
    print(np.linalg.solve(A_original, b_original))

    print("\n||Ax-b||2 =")
    print(np.linalg.norm(A_original @ x - b_original))


# ---------------- Example ----------------

A = np.array([[25,5,1],
              [64,8,1],
              [144,12,1]])

b = np.array([106.8,177.2,279.2])

gauss_elim(A,b)