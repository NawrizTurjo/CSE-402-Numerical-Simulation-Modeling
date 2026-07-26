"""
P2 -- Round-off Error & Complexity              [~25 min]  source: res/A2_Prep.md sec. 9
========================================================================================

QUESTION
--------
Given the system (whose exact solution is very close to x = [1, 1, 1]):
    20x1 +     15x2 + 10x3 = 45
    -3x1 - 2.2465x2 +  7x3 = 1.7505
     5x1 +      1x2 +  3x3 = 9

(a) Solve with NAIVE Gauss elimination (NO pivoting), simulating limited-
    precision hardware by rounding every intermediate value to 3 decimals.
(b) Solve the same system, same simulated precision, but WITH partial pivoting.
(c) Compare both against the true solution (np.linalg.solve on the un-rounded
    system) and report the max error for each.

WRITTEN QUESTION: for the SAME A but m different right-hand sides, state the
dominant-term operation count for (i) solving each from scratch with Gaussian
elimination vs (ii) decomposing A = LU once and reusing it for all m solves.

WHAT THIS DRILLS
----------------
- WHY partial pivoting exists. Not "in case of a zero pivot" -- that is the
  rare case. The real reason is that a SMALL pivot produces a HUGE multiplier,
  and a huge multiplier amplifies whatever round-off is already in the row.
- Reading a catastrophic answer that has no obvious symptom: no crash, no nan,
  no warning -- just a confidently printed wrong number.
"""
import numpy as np

DECIMALS = 3   # simulated limited precision


def r(value):
    """Round to the simulated machine precision -- applied to EVERY intermediate."""
    return round(value, DECIMALS)


def eliminate(Aug, use_pivoting, verbose=True):
    """
    Forward elimination at simulated precision, with pivoting optional so the
    two runs differ in exactly one line.
    """
    Aug = [row[:] for row in Aug]
    n = len(Aug)
    label = "pivoted" if use_pivoting else "naive"

    for k in range(n - 1):
        if use_pivoting:
            pivot_row = k
            for i in range(k + 1, n):
                if abs(Aug[i][k]) > abs(Aug[pivot_row][k]):
                    pivot_row = i
            if pivot_row != k:
                if verbose:
                    print(f"  [{label}] SWAP R{k + 1} <-> R{pivot_row + 1} "
                          f"(|{Aug[pivot_row][k]}| > |{Aug[k][k]}|)")
                Aug[k], Aug[pivot_row] = Aug[pivot_row], Aug[k]

        pivot = Aug[k][k]
        for i in range(k + 1, n):
            factor = r(Aug[i][k] / pivot)
            if verbose:
                print(f"  [{label}] step {k + 1}: pivot = {pivot}, "
                      f"multiplier for R{i + 1} = {factor}")
            for j in range(n + 1):
                Aug[i][j] = r(Aug[i][j] - factor * Aug[k][j])

        if verbose:
            print(f"  [{label}] after step {k + 1}:")
            for row in Aug:
                print("     ", row)
    return Aug


def back_substitution(Aug):
    """Back substitution, also at simulated precision."""
    n = len(Aug)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = Aug[i][n]
        for j in range(i + 1, n):
            total = r(total - Aug[i][j] * x[j])
        x[i] = r(total / Aug[i][i])
    return x


if __name__ == "__main__":
    A = [[20, 15, 10],
         [-3, -2.2465, 7],
         [5, 1, 3]]
    b = [45, 1.7505, 9]
    n = 3
    Aug0 = [A[i][:] + [b[i]] for i in range(n)]

    print("=" * 70)
    print(f"(a) NAIVE GAUSS ELIMINATION -- no pivoting, {DECIMALS}-decimal precision")
    print("=" * 70)
    x_naive = back_substitution(eliminate(Aug0, use_pivoting=False))
    print(f"\n  x_naive = {x_naive}")

    print("\n" + "=" * 70)
    print(f"(b) PARTIAL PIVOTING -- same {DECIMALS}-decimal precision")
    print("=" * 70)
    x_pivot = back_substitution(eliminate(Aug0, use_pivoting=True))
    print(f"\n  x_pivoted = {x_pivot}")

    print("\n" + "=" * 70)
    print("(c) COMPARISON AGAINST THE TRUE (UN-ROUNDED) SOLUTION")
    print("=" * 70)
    x_true = np.linalg.solve(np.array(A, dtype=float), np.array(b, dtype=float))

    err_naive = max(abs(x_naive[i] - x_true[i]) for i in range(n))
    err_pivot = max(abs(x_pivot[i] - x_true[i]) for i in range(n))

    print(f"  true solution (numpy, full precision) = {x_true}")
    print(f"  naive    = {x_naive}   max error = {err_naive:.4f}  "
          f"({100 * err_naive / max(abs(v) for v in x_true):.1f}% of the answer)")
    print(f"  pivoted  = {x_pivot}   max error = {err_pivot:.4f}")

    print("\n  WHY: naive elimination leaves 0.003 as the step-2 pivot, with")
    print("       -2.75 sitting right below it. Multiplier = -2.75/0.003 ~ -917,")
    print("       which multiplies the 3-decimal round-off already in that row")
    print("       by ~900x. Pivoting swaps -2.75 up instead, giving a multiplier")
    print("       of ~0.001 -- round-off gets SHRUNK rather than amplified.")
    print("\n  Note there is no crash, no nan and no warning in the naive run.")
    print("  A wrong answer here looks exactly like a right one.")

    assert err_pivot < 0.01, err_pivot
    assert err_naive > 0.1, err_naive     # the failure is the point of the exercise
    print("\n  ALL CHECKS PASSED (pivoted accurate, naive catastrophically off)")


# ----------------------------------------------------------------------
# WRITTEN QUESTION -- answer
#
# "For the same coefficient matrix A but m different right-hand sides, state
#  the dominant-term operation count for (i) Gaussian elimination redone per
#  right-hand side, vs (ii) A = LU decomposed once and reused."
#
# (i) GAUSSIAN ELIMINATION, REDONE PER RIGHT-HAND SIDE
#     Forward elimination on [A|b] is ~(2/3)n^3 flops; back substitution adds
#     ~n^2. Because the elimination is carried out on the AUGMENTED matrix, the
#     work is entangled with that particular b and nothing survives for the
#     next one -- so it is repeated in full:
#
#         total ~ m * (2/3)n^3  =  O(m n^3)
#
# (ii) LU ONCE, REUSED m TIMES
#     The factorization touches only A, so it happens exactly once at
#     ~(2/3)n^3. Each right-hand side then costs forward substitution (~n^2)
#     plus backward substitution (~n^2):
#
#         total ~ (2/3)n^3 + m * 2n^2  =  O(n^3 + m n^2)
#
# For m = 1 the two are the same (LU is doing identical arithmetic, just
# bookkept differently). For every m > 1, (ii) wins, and the gap grows
# linearly in m: at n = 100, m = 50, that is roughly 33 million flops versus
# 1.7 billion -- a ~50x difference from the same arithmetic, organized once
# instead of fifty times.
#
# This is the same conclusion as C2's written question, stated as a complexity
# bound rather than a runtime observation, and P1b's inverse is the special
# case m = n with the right-hand sides being the columns of I.
# ----------------------------------------------------------------------
