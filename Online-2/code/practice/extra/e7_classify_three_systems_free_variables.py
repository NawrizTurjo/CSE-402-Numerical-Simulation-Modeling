"""
E7 -- Classifying Systems, Including TWO Free Variables        [~35 min]
========================================================================
Topics: Gauss elimination, rank, unique / no-solution / infinite classification,
        parametrizing a solution set with more than one free variable.
Closest real question: B1 (three systems, classify each). Difficulty: core, harder variant.

QUESTION
--------
For each of the following 4x4 systems:

  System 1 (A1, b1):  [[1,2,-1,3],[2,4,1,0],[-1,1,2,1],[3,1,0,2]]   b=[8,11,3,7]
  System 2 (A2, b2):  [[1,2,3,4],[2,4,6,8],[1,1,1,1],[3,3,3,3]]     b=[10,20,4,12]
  System 3 (A3, b3):  [[1,2,3,4],[2,4,6,8],[1,1,1,1],[3,3,3,3]]     b=[10,21,4,12]

(a) Run Gaussian elimination with partial pivoting, hand-written. Print the
    pivot chosen and any swap per column, and the augmented matrix after each.
(b) Classify from the reduced form:
      a row  0 0 0 0 | c  with c != 0  -> NO SOLUTION (print the bad equation)
      a row  0 0 0 0 | 0               -> INFINITELY MANY (one free variable
                                           per missing pivot)
      otherwise                        -> UNIQUE
(c) Act on the classification:
      unique   -> back-substitute, verify against np.linalg.solve, report
                  ||Ax-b||_2.
      infinite -> identify WHICH variables are free, pick values for them, and
                  produce one particular solution; verify it satisfies the
                  ORIGINAL system.
      no soln  -> report the contradiction; do not attempt to solve.
(d) Cross-check every verdict with the rank test:
      rank(A) < rank([A|b])            -> no solution
      rank(A) == rank([A|b]) == n      -> unique
      rank(A) == rank([A|b]) <  n      -> infinite

WRITTEN QUESTION: System 2 has TWO free variables. How do you know the number
of free variables before choosing any of them, and how many solutions does that
give?

WHY THIS IS WORTH DRILLING
--------------------------
B1's real question was exactly this, at 3x3 with at most one free variable.
Systems 2 and 3 here are rank 2 out of 4, so there are two free variables and
the solution set is a plane, not a line -- the case where a hand-written
classifier that assumes "one free variable" quietly returns nonsense. Systems 2
and 3 also differ in a SINGLE entry of b, flipping infinite -> no solution,
which is the cleanest possible demonstration that the RHS alone decides.
"""
import numpy as np

EPS = 1e-9


def forward_elimination(A, b, verbose=True):
    """
    Elimination with partial pivoting, tracking which column each pivot landed
    in. A column with no usable pivot is a FREE column and the row cursor does
    not advance -- that is what generalizes B1's 3x3 code to rank < n - 1.
    """
    n = len(b)
    Aug = [list(A[i]) + [b[i]] for i in range(n)]
    pivot_columns = []
    row = 0

    for col in range(n):
        if row >= n:
            break

        pivot_row = row
        for i in range(row + 1, n):
            if abs(Aug[i][col]) > abs(Aug[pivot_row][col]):
                pivot_row = i

        if verbose:
            print(f"\n  --- column {col + 1} ---")
            print(f"    candidates (rows {row + 1}..{n}): "
                  f"{[round(Aug[i][col], 6) for i in range(row, n)]}")

        if abs(Aug[pivot_row][col]) < EPS:
            if verbose:
                print(f"    no usable pivot -> x{col + 1} is a FREE VARIABLE, "
                      "row cursor stays put")
            continue

        if pivot_row != row:
            if verbose:
                print(f"    SWAP R{row + 1} <-> R{pivot_row + 1}")
            Aug[row], Aug[pivot_row] = Aug[pivot_row], Aug[row]

        pivot = Aug[row][col]
        if verbose:
            print(f"    pivot = {pivot:.6g} at (row {row + 1}, col {col + 1})")

        for i in range(row + 1, n):
            factor = Aug[i][col] / pivot
            for j in range(col, n + 1):
                Aug[i][j] -= factor * Aug[row][j]

        pivot_columns.append(col)
        row += 1

        if verbose:
            for r in Aug:
                print("     ", [round(value, 6) for value in r])

    return Aug, pivot_columns


def classify(Aug, pivot_columns, verbose=True):
    """
    Reads the verdict off the reduced augmented matrix.

    Returns (kind, free_columns, bad_row_index).
    """
    n = len(Aug)
    free_columns = [col for col in range(n) if col not in pivot_columns]

    for i in range(n):
        row_is_zero = all(abs(Aug[i][j]) < EPS for j in range(n))
        if row_is_zero and abs(Aug[i][n]) > EPS:
            if verbose:
                print(f"\n  row {i + 1} reduced to: "
                      f"{' + '.join('0*x%d' % (j + 1) for j in range(n))} "
                      f"= {Aug[i][n]:.6g}   -> CONTRADICTION")
            return "no_solution", free_columns, i

    if free_columns:
        if verbose:
            zero_rows = [i + 1 for i in range(n)
                         if all(abs(Aug[i][j]) < EPS for j in range(n + 1))]
            print(f"\n  rank = {len(pivot_columns)} < n = {n}")
            print(f"  rows reduced to 0 = 0: {zero_rows}")
            print(f"  free variables: {['x%d' % (c + 1) for c in free_columns]}"
                  f"  -> INFINITELY MANY SOLUTIONS")
        return "infinite", free_columns, None

    if verbose:
        print(f"\n  rank = {len(pivot_columns)} = n -> every pivot present "
              "-> UNIQUE SOLUTION")
    return "unique", [], None


def back_substitute(Aug, pivot_columns, free_values):
    """
    Back-substitutes with the free variables pinned to chosen values.

    Works for any number of free variables: walk the pivot rows from the bottom
    up, and every already-assigned variable to the right (pivot or free) just
    moves to the right-hand side.
    """
    n = len(Aug)
    x = [0.0] * n
    for col, value in free_values.items():
        x[col] = value

    for row in range(len(pivot_columns) - 1, -1, -1):
        col = pivot_columns[row]
        total = Aug[row][n]
        for j in range(col + 1, n):
            total -= Aug[row][j] * x[j]
        x[col] = total / Aug[row][col]
    return x


def matvec(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def l2_norm(vec):
    total = 0.0
    for value in vec:
        total += value * value
    return total ** 0.5


def solve_and_report(name, A, b, free_choice):
    """Runs the whole pipeline on one system and verifies the verdict."""
    n = len(b)
    print("\n" + "=" * 70)
    print(f"{name}")
    print("=" * 70)
    for i in range(n):
        print(f"  {A[i]} | {b[i]}")

    Aug, pivot_columns = forward_elimination(A, b)
    kind, free_columns, bad_row = classify(Aug, pivot_columns)

    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)

    # (d) independent second opinion from the rank test
    rank_A = int(np.linalg.matrix_rank(A_np))
    rank_aug = int(np.linalg.matrix_rank(np.hstack([A_np, b_np.reshape(-1, 1)])))
    if rank_A < rank_aug:
        expected = "no_solution"
    elif rank_A == n:
        expected = "unique"
    else:
        expected = "infinite"
    print(f"\n  RANK TEST: rank(A) = {rank_A}, rank([A|b]) = {rank_aug}, n = {n}"
          f"  -> '{expected}'")
    print(f"  our elimination said: '{kind}'   "
          f"[{'AGREE' if kind == expected else 'DISAGREE -- BUG'}]")
    assert kind == expected

    if kind == "no_solution":
        print(f"\n  Nothing to solve. Row {bad_row + 1} says 0 = "
              f"{Aug[bad_row][n]:.6g}, which no x can satisfy.")
        return kind

    if kind == "unique":
        x = back_substitute(Aug, pivot_columns, {})
        print(f"\n  x = {[round(value, 8) for value in x]}")
    else:
        print(f"\n  choosing free variables: "
              f"{', '.join('x%d = %g' % (c + 1, free_choice[c]) for c in free_columns)}")
        x = back_substitute(Aug, pivot_columns, free_choice)
        print(f"  one particular solution: x = {[round(value, 8) for value in x]}")

    # residual by hand first, numpy after -- against the ORIGINAL system
    Ax = matvec(A, x)
    residual = [Ax[i] - b[i] for i in range(n)]
    manual_res = l2_norm(residual)
    numpy_res = float(np.linalg.norm(A_np @ np.array(x) - b_np))
    print(f"  Ax - b = {[round(value, 12) for value in residual]}")
    print(f"  manual ||Ax-b||_2 = {manual_res:.3e}   numpy = {numpy_res:.3e}")
    assert manual_res < 1e-8

    if kind == "unique":
        x_lib = np.linalg.solve(A_np, b_np)
        print(f"  np.linalg.solve = {np.round(x_lib, 8)}   "
              f"gap = {np.linalg.norm(np.array(x) - x_lib):.3e}")
        assert np.allclose(x, x_lib, atol=1e-8)
    else:
        # a second choice of free values must ALSO satisfy the system --
        # that is what "infinitely many" means, demonstrated rather than asserted
        other = {c: free_choice[c] + 3.0 for c in free_columns}
        x2 = back_substitute(Aug, pivot_columns, other)
        res2 = l2_norm([matvec(A, x2)[i] - b[i] for i in range(n)])
        print(f"\n  a DIFFERENT choice "
              f"({', '.join('x%d = %g' % (c + 1, other[c]) for c in free_columns)}):")
        print(f"    x = {[round(value, 8) for value in x2]}   "
              f"||Ax-b||_2 = {res2:.3e}")
        print("    -> also an exact solution, which is the point: the solution")
        print(f"       set is a {len(free_columns)}-parameter family")
        assert res2 < 1e-8
        print("  (np.linalg.solve is deliberately NOT called here -- it raises "
              "on a singular A)")

    return kind


if __name__ == "__main__":
    A1 = [[1, 2, -1, 3],
          [2, 4, 1, 0],
          [-1, 1, 2, 1],
          [3, 1, 0, 2]]
    b1 = [8, 11, 3, 7]

    # rows 2 and 4 are exact multiples of rows 1 and 3 -> rank 2 out of 4
    A2 = [[1, 2, 3, 4],
          [2, 4, 6, 8],
          [1, 1, 1, 1],
          [3, 3, 3, 3]]
    b2 = [10, 20, 4, 12]      # consistent: 20 = 2*10 and 12 = 3*4
    b3 = [10, 21, 4, 12]      # ONE entry changed: 21 != 2*10 -> contradiction

    kinds = []
    kinds.append(solve_and_report("SYSTEM 1 -- expected: unique", A1, b1, {}))
    kinds.append(solve_and_report("SYSTEM 2 -- expected: infinite (2 free vars)",
                                  A2, b2, {2: 1.0, 3: 0.0}))
    kinds.append(solve_and_report("SYSTEM 3 -- expected: no solution",
                                  A2, b3, {}))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, kind in zip(["System 1", "System 2", "System 3"], kinds):
        print(f"  {name}: {kind}")
    print("\n  Systems 2 and 3 share the SAME matrix and differ in one entry of")
    print("  b (20 vs 21). The coefficient matrix decides that SOME rows will")
    print("  collapse; the right-hand side alone decides whether each collapsed")
    print("  row reads 0 = 0 (redundant) or 0 = 1 (contradictory).")

    assert kinds == ["unique", "infinite", "no_solution"], kinds
    print("\n  ALL CHECKS PASSED")


# ----------------------------------------------------------------------
# WRITTEN QUESTION -- answer
#
# "System 2 has TWO free variables. How do you know the number of free
#  variables before choosing any of them, and how many solutions does that
#  give?"
#
# COUNTING THEM. The count is fixed by the RANK, and you get it for free from
# the elimination you already ran:
#
#     number of free variables = n - rank(A)
#
# where rank(A) is simply how many pivots the elimination actually placed. You
# do not need to guess or inspect: every column that failed to receive a pivot
# is a free column, which is exactly what `pivot_columns` records above.
#
# For System 2, n = 4 and only two pivots land (row 2 is 2x row 1, row 4 is
# 3x row 3, so two rows collapse to all zeros), giving rank(A) = 2 and
# 4 - 2 = 2 free variables. B1's 3x3 systems had rank 2 out of 3, hence exactly
# one free variable -- which is why code written for B1 that assumes a single
# free variable silently breaks here.
#
# HOW MANY SOLUTIONS. Assuming the system is consistent (rank(A) ==
# rank([A|b]), otherwise there are none at all), each free variable can take
# ANY real value independently, and the pivot variables are then determined by
# back substitution. So the solution set is a k-parameter family with
# k = n - rank(A):
#
#     k = 0  ->  a single point           (the unique solution)
#     k = 1  ->  a line in R^n            (B1's case)
#     k = 2  ->  a plane in R^n           (System 2 here)
#
# Formally, the solution set is  x = x_particular + span{h_1, ..., h_k}, where
# the h_i span the null space of A. Setting free variable i to 1 and the rest
# to 0 (and back-substituting with b replaced by 0) produces h_i directly.
# The demonstration in the code -- shifting the free values by 3 and getting a
# second exact solution -- is that structure showing up concretely.
#
# THE CONSISTENCY CAVEAT, which is the whole point of System 3: rank(A) = 2 in
# BOTH System 2 and System 3, so both have two would-be free variables. But
# System 3 has rank([A|b]) = 3 > rank(A) = 2, so one collapsed row reads 0 = 1
# and the family is empty. Free variables only count once consistency has been
# established -- "rank-deficient" means "no unique solution", not "infinitely
# many solutions".
# ----------------------------------------------------------------------
