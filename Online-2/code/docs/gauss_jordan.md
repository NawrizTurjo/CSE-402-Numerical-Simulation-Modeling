# Gauss-Jordan Elimination

Library: [`algorithms/gauss_jordan.py`](../algorithms/gauss_jordan.py)
Exam template: [`templates/02_gauss_jordan.py`](../templates/02_gauss_jordan.py)

## Algorithm level

**Goal:** same as Gauss elimination (`Ax=b`), but drive `[A|b]` all the way
to `[I|x]` — the solution sits directly in the last column, no back
substitution needed.

Two upgrades over plain Gauss elimination, applied at every pivot step `k`:
1. **Normalize** the pivot row so the pivot becomes exactly 1:
   `row_k <- row_k / A[k,k]`.
2. **Eliminate the pivot variable from every OTHER row** — not just the
   ones below (as in Gauss elimination), but the ones *above* too:
   ```
   for i != k:  row_i <- row_i - A[i,k] * row_k
   ```

After all `n` columns are cleared this way, the left block is the identity
and the right column is `x`.

**Cost vs. Gauss elimination:** strictly more arithmetic (every row is
touched at every step, not just the rows below the pivot), but the payoff
is skipping the separate back-substitution pass. Same pitfalls
(division-by-zero, round-off) and the same fix (partial pivoting) carry
over unchanged from Gauss elimination.

## Code level

- `Aug = np.hstack([A, b.reshape(-1,1)])` builds the augmented matrix once;
  every operation afterward touches `Aug` directly, including the RHS
  column automatically (no separate bookkeeping for `b` needed, unlike
  Gauss elimination where `A` and `b` are updated in lockstep as two
  separate arrays).
- The elimination loop `for i in range(n): if i != k: ...` is the one
  structural difference from Gauss elimination's `for i in range(k+1, n)`
  — that's the entire "eliminate above and below" upgrade in code form.
- Partial pivoting (`partial_pivot=True` default) works identically to
  Gauss elimination: largest `|entry|` in column `k` at/below row `k`.
- **Gotcha:** normalizing (`Aug[k] = Aug[k] / Aug[k,k]`) must happen
  *before* the elimination loop that references `Aug[k]` as the row being
  subtracted — otherwise you're eliminating with the un-normalized pivot
  row and the identity never fully forms.
- No separate `classify_and_solve` here — Gauss-Jordan's zero-row logic
  for no-solution/infinite-solution cases is identical to Gauss
  elimination's (see `docs/gauss_elimination.md`); if a question needs
  that classification alongside Gauss-Jordan style output, reuse
  `algorithms/gauss_elimination.py`'s `classify_and_solve` for the
  decision logic and this file's normalize-and-clear-every-row loop for
  the presentation.

## Common written questions

- **Why does Gauss-Jordan need no back substitution?** Because it clears
  the pivot variable from *every* other equation as it goes, not just the
  ones below — by the time all `n` columns are processed, each row already
  contains only its own variable's contribution.
- **Is it ever worth using over Gauss elimination?** Rarely for solving a
  single `Ax=b` (more arithmetic for the same answer) — its real use case
  is computing a matrix inverse in one pass (`[A|I] -> [I|A^-1]`), where
  "no separate back-substitution per column" actually saves work.
