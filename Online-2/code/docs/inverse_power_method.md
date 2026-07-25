# Inverse Power Method

Library: [`algorithms/inverse_power_method.py`](../algorithms/inverse_power_method.py)
Exam template: [`templates/05_inverse_power_method.py`](../templates/05_inverse_power_method.py)

## Algorithm level

**Goal:** find the **smallest**-magnitude eigenvalue of `A`.

**Key fact:** if `lambda` is an eigenvalue of `A`, then `1/lambda` is an
eigenvalue of `A^-1` (one-line proof: `Av = lam*v` ⟹ `v = lam*A^-1 v` ⟹
`A^-1 v = (1/lam) v`). Inversion **flips the ranking** — the smallest
eigenvalue of `A` becomes the *largest* (dominant) eigenvalue of `A^-1`.

So: run the ordinary power method on `A^-1` instead of `A`. It converges to
`1/lambda_min`; take the reciprocal at the end to recover `lambda_min`.

**Steps:**
1. Start with any nonzero `x0`.
2. Each iteration, compute `y_{k+1} = A^-1 x_k` — in practice, **solve**
   `A y_{k+1} = x_k` instead of ever forming `A^-1` (via LU: factor `A=LU`
   once, then each iteration is `Lz=x_k` forward-solve + `Uy=z`
   back-solve).
3. Normalize `y_{k+1}` by its largest-magnitude entry to get `x_{k+1}`;
   that normalizing factor is the current estimate of `1/lambda_min`.
4. Repeat until convergence; report `lambda_min = 1 / (final factor)`.

## Code level

- `inverse_power_naive` calls `np.linalg.inv(A)` **once**, then runs
  ordinary power iteration on the resulting matrix — simple, and fine if
  the question explicitly permits/asks for `np.linalg.inv()` (the actual
  past A1 exam question did name it directly — read the exact wording on
  the day).
- `inverse_power_via_lu` is the version worth defaulting to otherwise: `A`
  is factored into `P, L, U` **once** outside the iteration loop, and every
  iteration is just two triangular solves (`_fsub`, `_bsub`) — `A^-1` is
  never materialized.
- Both return `1.0 / history[-1]` — remember the reciprocal step; forgetting
  it is a very easy last-line bug (you'd silently report `1/lambda_min`
  instead of `lambda_min`).
- `_plu`/`_fsub`/`_bsub` here are trimmed private copies of the functions in
  `algorithms/lu_decomposition.py` (kept local so this file is a single
  self-contained copy-paste unit — the same duplication choice used
  throughout `algorithms/` and `templates/`).

## Anti-patterns seen in the raw collected code (don't do this)

`res/Src-26/inversepowermethod.py` is worth a look specifically because it
fails in two ways that are easy to miss by eye:
- `eig = np.max(np.abs(y))` throws away the **sign** entirely (`np.max` of
  an already-`abs`'d array is always positive) — different from the more
  common `np.max(y)` bug, but the same root problem: the reported
  eigenvalue is silently wrong-signed whenever the true dominant entry is
  negative. Always use `y[np.argmax(np.abs(y))]` — magnitude picks *which*
  entry, plain indexing preserves *its* sign.
- It recurses one call per iteration and returns
  `(1/eig, invpowermethod(...))` — a tuple whose second element is another
  such tuple, nested `max_iter` levels deep, terminating in `None`. The
  **converged** answer ends up buried at the bottom of that nested
  structure; the value printed at the top level is actually the
  **first** (least accurate) iteration's estimate, not the converged one.
  Prefer an explicit loop that appends to a `history` list and returns the
  last element once a convergence check passes (as every function in
  `algorithms/` does) — it's both correct and inspectable.

## Common written questions

- **Both "solve via LU" and "multiply by precomputed `A^-1`" are `O(n^2)`
  per iteration — so why is the LU version still better?** Two reasons:
  (1) **setup cost** — forming `A^-1` costs `~2n^3` (n solves) vs. `~2n^3/3`
  for one LU factorization, 3x cheaper setup. (2) **sparsity** — if `A` is
  banded/block-sparse, `L` and `U` preserve that structure (triangular
  solves stay cheap), but `A^-1` of a sparse matrix is generally **dense**
  — for large sparse systems this is the difference between feasible and
  infeasible.
- **What if `A` is singular or nearly singular?** The inverse power method
  breaks down exactly there — `A^-1` doesn't exist (or is numerically
  huge/unstable). See `complex_cases/singular_and_rank_deficient.py`.
