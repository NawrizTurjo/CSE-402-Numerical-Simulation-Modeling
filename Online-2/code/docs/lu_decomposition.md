# LU Decomposition

Library: [`algorithms/lu_decomposition.py`](../algorithms/lu_decomposition.py)
Exam template: [`templates/03_lu_decomposition.py`](../templates/03_lu_decomposition.py)

## Algorithm level

**Goal:** factor `A = LU` once (`L` unit-lower-triangular, `U`
upper-triangular), then reuse the factors to solve `Ax=b` for as many `b`
as needed, cheaply.

**Where `U` comes from:** exactly the result of Gauss forward elimination
on `A` — nothing new to compute.

**Where `L` comes from:** the multipliers `m_ik = a_ik^(k-1) / a_kk^(k-1)`
that Gauss elimination computes at every step and normally throws away.
Store `m_ik` directly at position `(i,k)` of `L` (with 1s on the
diagonal). This is *exactly correct* because of the elementary-matrix
view: each elimination step is left-multiplication by an elementary matrix
`E_k`; applying all of them gives `E_n...E_1 A = U`, so
`A = (E_1^-1...E_n^-1) U`. Each `E_k^-1` just flips its multiplier's sign,
and — because elimination proceeds column by column with multipliers
landing in slots that never interact — the product of all the inverses
collapses into a single unit-lower-triangular matrix with the multipliers
sitting exactly where they were computed. No matrix inversion or
multiplication is actually needed at runtime.

**Existence.** `A = LU` (no pivoting) exists iff every **leading principal
minor** of `A` is nonzero — NOT the same as `det(A) != 0`. Counterexample:
`[[0,1],[1,0]]` (det = -1, invertible, but the 1x1 leading minor is 0, so
no LU without pivoting). **`PA = LU`** (partial pivoting) exists for
**every invertible matrix** — a valid pivot is always available at every
step, because a fully-zero column at and below the diagonal would make `A`
singular, contradicting invertibility.

**Solving `Ax=b` via LU:** introduce `z = Ux`, split into two triangular
solves:
```
Lz = b   (forward substitution, top to bottom)
Ux = z   (back substitution, bottom to top)
```
With pivoting, solve `Lz = P@b`, **not** `Lz = b` — the permutation must be
applied to `b` too.

**Why bother — cost.** Factoring costs `O(n^3)` (~`2n^3/3` flops), same as
one full Gauss elimination — for a *single* solve, LU buys nothing. The
payoff appears with **multiple right-hand sides, same `A`**: factor once,
then each new `b` costs only `O(n^2)` (two triangular solves) instead of
redoing the full `O(n^3)` elimination. For `k` right-hand sides:
`O(n^3 + k*n^2)` (LU) vs. `O(k*n^3)` (repeated Gauss) — at `n=10000` this
is roughly a 2500x difference for computing a full inverse (`k=n`).

**Determinant:** `PA=LU` → `det(P)*det(A) = det(L)*det(U)`. `det(L)=1`
(unit diagonal), `det(P) = (-1)^swaps` and is its own inverse, so:
```
det(A) = (-1)^(#swaps) * product(diagonal of U)
```

**Inverse:** column `j` of `A^-1` solves `A x_j = e_j` (`e_j` = column `j`
of the identity). Factor once, then for each `j`: forward-solve
`Lz = P@e_j`, back-solve `Ux_j = z`. **Never** form `A^-1` just to solve
`Ax=b` — a direct solve costs `~2n^2` flops with the factors in hand;
forming `A^-1` first costs `n` solves `~2n^3` flops, `n` times more
expensive for the same answer, and compounds rounding error over `n`
solves instead of one.

**Cholesky (`A = LL^T`)** replaces general LU when `A` is symmetric
positive-definite: about half the cost (`~n^3/3` flops, half the storage),
and needs **no pivoting** (SPD guarantees every pivot stays positive).

## Code level

- `plu_decomposition` is the one function almost everything else in this
  file builds on — `lu_determinant`, `lu_inverse`, and `solve_via_lu` all
  call it once and then just do triangular solves.
- **Critical detail easy to get wrong:** when a row swap happens mid-
  factorization, the multipliers *already stored* in `L`'s earlier columns
  must move with their row too:
  ```python
  if k > 0:
      L[[k, p], :k] = L[[p, k], :k]
  ```
  Skip this and `L` describes an elimination history that never actually
  happened — `P@A` will NOT equal `L@U` even though the code runs without
  error.
- `forward_substitution`/`backward_substitution` are shared by every
  higher-level function (`solve_via_lu`, `lu_inverse`) — get these two
  right once and everything downstream is just bookkeeping.
- `lu_naive` (no pivoting) is included specifically to demonstrate
  *failure*: run it on a matrix with a zero leading principal minor (e.g.
  `[[0,2,1],[1,1,1],[2,1,3]]`) and watch `L`/`U` fill with `inf`/`nan` —
  NumPy does not raise an exception for `x/0` or `inf*0`, it silently
  produces `inf` then `nan`. See
  `complex_cases/singular_and_rank_deficient.py` for this played out with
  printed warnings.
- **Gotcha:** `lu_inverse`'s loop assigns `Ainv[:, j] = x` directly — `x`
  IS column `j` already (forward+back substitution returns a 1-D vector
  matching that shape), so no `.T` is needed here. (Contrast with code that
  builds a result matrix row-by-row and needs `.T` at the end — a classic
  bug noted in `res/Src-14/lu_decomposition_theory.md` section 13.)

## Common written questions

- **Cost of factor vs. one solve?** Factor `O(n^3)` once; each
  forward+backward solve pair `O(n^2)` (~`2n^2` flops).
- **When would you use Cholesky instead of general LU?** `A` symmetric
  positive-definite → half the cost, no pivoting needed.
- **Small `||Ax-b||` but "wrong" `x`?** Ill-conditioning — check
  `kappa(A) = ||A||*||A^-1||`, not just the residual;
  `relative error in x <~ kappa(A) * relative residual`. See
  `complex_cases/ill_conditioned_systems.py`.
