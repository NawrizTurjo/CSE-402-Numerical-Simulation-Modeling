# Gauss Elimination

Library: [`algorithms/gauss_elimination.py`](../algorithms/gauss_elimination.py)
Exam template: [`templates/01_gauss_elimination.py`](../templates/01_gauss_elimination.py)

## Algorithm level

**Goal:** solve `Ax = b`, `n` equations in `n` unknowns, by reducing `[A|b]`
to upper-triangular form and back-substituting.

**Forward elimination.** For each pivot step `k = 0 .. n-2`:
1. Choose a pivot row: the row `p >= k` with the largest `|A[p,k]|`
   (partial pivoting).
2. Swap rows `k` and `p` (in both `A` and `b`).
3. For every row `i > k`, compute the multiplier `m = A[i,k] / A[k,k]` and
   subtract `m` times row `k` from row `i` (this zeroes `A[i,k]`):
   ```
   row_i <- row_i - m * row_k         (applied to both A and b)
   ```

After `n-1` steps, `A` is upper-triangular.

**Back substitution.** Starting from the last row (one unknown) and working
up:
```
x_i = (c_i - sum_{j>i} U[i,j]*x_j) / U[i,i]
```
Each `x_j` used in the sum is already known from a later row.

**Why partial pivoting?** Two independent failure modes without it:
- A pivot of exactly 0 → division by zero (the equation isn't unsolvable,
  the row order is just unlucky).
- A pivot that's merely *small* → a huge multiplier → catastrophic
  cancellation amplifies rounding error (worked example in
  `Slides/3_system_of_eqs.txt` shows a 5-digit vs 6-digit run of the SAME
  system diverging to 50% vs 5% error from one tiny pivot). Partial
  pivoting bounds every multiplier at `|m| <= 1`.

**Classifying the solution** (after elimination, look at each row):

| Row after elimination | Meaning | Result |
|---|---|---|
| No fully-zero coefficient row | full rank | Unique solution |
| `0 = c`, `c != 0` | contradiction | No solution |
| `0 = 0` | redundant equation | Infinite solutions (free variable) |

A zero **diagonal** entry alone does NOT mean no solution — only a fully
zero **row** (after all pivoting has been attempted) does. See
`res/Src-2/q2_written_answer.md` for the full worked answer to this exact
past written question.

**Determinant for free:** row operations (adding a multiple of one row to
another) don't change `det`; a row **swap** flips its sign; a triangular
matrix's determinant is the product of its diagonal. So:
```
det(A) = (-1)^(number of swaps) * product(diagonal of U)
```

## Code level

- `forward_elimination(A, b, hand_written=True)` — the `hand_written` flag
  toggles between an explicit double-loop pivot search/swap (what past
  exams have explicitly required — no `np.argmax`/fancy indexing) and the
  vectorized one-liner equivalent. Keep `hand_written=True` unless the
  question says otherwise.
- `A[i, k:] -= m * A[k, k:]` is the entire inner elimination loop —
  columns before `k` are already zero in both rows so there's nothing to
  update there.
- `classify_and_solve` generalizes past exams' single-free-variable case to
  **any number of simultaneous free variables** (rank deficiency of any
  degree). The trick: with standard `n x n` partial-pivoting elimination,
  elimination step `k` always targets column `k`. If an entire zero row
  appears at index `i`, that row structurally corresponds to variable `x_i`
  being free — so iterating back-substitution from `i=n-1` down to `0`,
  assigning `free_value` at every zero row and solving normally otherwise,
  correctly handles multiple free variables in one pass (every `x_j` a row
  needs, `j > i`, is already resolved by the time row `i` is processed).
  See `complex_cases/multiple_free_variables.py` for a rank-deficiency-2
  worked example.
- **Gotcha:** `EPS = 1e-9` is the "is this basically zero" threshold —
  tune it if your matrix has entries at wildly different scales (very
  large or very small numbers can make a genuinely-zero row read as
  slightly nonzero, or vice versa).
- `determinant()` re-derives its own elimination rather than reusing
  `forward_elimination`, because it needs to bail out early (return `0.0`)
  the instant a pivot is exactly zero — a zero row later in
  `classify_and_solve` is a *feature* there (it signals no/infinite
  solutions), but here it just means `det = 0` immediately.

## Common written questions

- **Why doesn't a zero diagonal entry always mean "no solution"?** Two
  cases: (1) some row *below* still has a nonzero entry in that column →
  pivoting swaps it up, elimination continues, unique solution still
  possible. (2) The pivot *and* everything below it in that column is also
  zero → singular; then the RHS of the resulting zero row decides
  no-solution (`c≠0`) vs. infinite solutions (`c=0`).
- **Cost of Gauss elimination?** `O(n^3)` for forward elimination
  (dominant term `~2n^3/3` flops), `O(n^2)` for back substitution.
