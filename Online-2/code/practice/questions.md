# Practice questions — statements only

**Solutions are deliberately not here.** Pick one, set a 30-minute timer, solve
it in a blank editor, and only then open the matching file in
[`a2prep/`](a2prep/) or [`extra/`](extra/) to compare.

Requirements that apply to **every** question below, because they applied to
every real past question:

- Hand-write the numerical method — pivot selection, row swaps, elimination,
  forward/back substitution, the iteration loop, normalization.
- Hand-write anything that is part of your own answer, including norms,
  residuals, and matrix-vector products. NumPy comes *after*, to re-derive the
  same number as a cross-check. Print both.
- Print every intermediate step: pivots chosen, row swaps, the matrix after
  each elimination step, L and U after each column, iteration-by-iteration
  eigenvalue estimates.
- Finish with a NumPy verification and an error/residual norm.
- Answer the written question in a comment, even briefly.

---

## P1a — Gauss-Jordan RREF & determinant `[~25 min]`

Given
```
 0x₁ +  x₂ + 2x₃ =  8
  x₁ -  x₂ + 3x₃ =  8
 2x₁ + 4x₂ +  x₃ = 13
```
i.e. `A = [[0,1,2],[1,-1,3],[2,4,1]]`, `b = [8,8,13]`.

**(a)** Reduce `[A|b]` to Reduced Row Echelon Form using Gauss-Jordan with
partial pivoting. Print per column: candidate pivots, pivot chosen, every row
swap, and the matrix after that column is cleared both **above** and **below**
the pivot. Read `x` directly off the final `[I|x]` — no back substitution.

**(b)** Using the same elimination, compute `det(A)` as a byproduct: the
product of the **raw** pivots (before each row is normalized), with a sign flip
per row swap.

**(c)** Verify against `np.linalg.solve(A,b)` and `np.linalg.det(A)`.

**Written:** why must the determinant come from the pivots as they were
*before* normalization, rather than from the fully-reduced RREF?

---

## P1b — Matrix inverse via Gauss-Jordan and via LU `[~30 min]`

Using `A = [[0,1,2],[1,-1,3],[2,4,1]]`:

**(a)** Compute `A⁻¹` by Gauss-Jordan on `[A | I]`.
**(b)** Compute `A⁻¹` by LU: factor `A` once, then solve `A xⱼ = eⱼ` for each
identity column, assembling the `xⱼ` as the columns of `A⁻¹`.
**(c)** Verify both agree with each other and with `np.linalg.inv(A)`.

**Written:** for a 1000×1000 matrix, which method would you prefer, and why?

---

## P2 — Round-off error & complexity `[~25 min]`

Given (exact solution very close to `x = [1,1,1]`):
```
 20x₁ +      15x₂ + 10x₃ = 45
 -3x₁ - 2.2465x₂ +  7x₃ = 1.7505
  5x₁ +       1x₂ +  3x₃ = 9
```

**(a)** Solve with **naive** Gauss elimination (no pivoting), rounding every
intermediate value to 3 decimals to simulate limited precision.
**(b)** Solve again at the same precision but **with** partial pivoting.
**(c)** Compare both against `np.linalg.solve` on the un-rounded system and
report the max error for each.

**Written:** for the same `A` but `m` different right-hand sides, state the
dominant-term operation count for (i) Gaussian elimination redone per RHS vs
(ii) `A = LU` decomposed once and reused.

---

## P3 — Direct eigenvalues via the characteristic polynomial `[~25 min]`

Given `A = [[6,2],[2,3]]`:

**(a)** Derive the characteristic polynomial `det(A - λI) = 0` in terms of
trace and determinant.
**(b)** Solve for both eigenvalues with the quadratic formula.
**(c)** For each eigenvalue, hand-solve `(A - λI)v = 0` for its eigenvector and
normalize to unit length.
**(d)** Verify against `np.linalg.eig(A)`.

---

## P4 — Full eigendecomposition & matrix powers `[~30 min]`

Using `A = [[6,2],[2,3]]` and P3's eigenpairs (`λ₁=7, v₁=[2/√5, 1/√5]`;
`λ₂=2, v₂=[1/√5, -2/√5]`):

**(a)** Build `V = [v₁ v₂]` (eigenvectors as **columns**) and `Λ = diag(λ₁,λ₂)`.
**(b)** Verify `A = VΛV⁻¹`.
**(c)** Take `x = [3,1]`. Convert to eigen-coordinates via `V⁻¹x`, then back via
`V(...)`, confirming you recover `x`.
**(d)** Compute `A⁵` via `VΛ⁵V⁻¹` and via `np.linalg.matrix_power(A,5)`;
confirm they agree.

---

## P5 — Deflation for intermediate eigenvalues `[~30 min]`

Given `A = [[12,2,1,0],[2,5,0,1],[1,0,3,1],[0,1,1,2]]`:

**(a)** Run the power method on `A` for the dominant eigenpair `(λ₁, v₁)`,
`v₁` normalized to unit length.
**(b)** Build the deflated matrix `A₂ = A - λ₁ v̂₁v̂₁ᵀ`.
**(c)** Run the power method again on `A₂` to get `(λ₂, v₂)` — the
second-largest eigenvalue of the **original** `A`.
**(d)** Verify `A₂v₁ ≈ 0` and `v₁·v₂ ≈ 0`.

**Written:** this trick relies on `A` being symmetric. Why — what specifically
would break if it weren't?

---

## E1 — Gauss elimination on a 4×4, with the determinant `[~25 min]`

```
A = [[0,  2,  1, -1],        b = [8, 3, 0, 19]
     [3, -1,  2,  4],
     [1,  5, -3,  2],
     [2,  1,  4, -3]]
```

**(a)** Solve `Ax = b` by Gaussian elimination with partial pivoting. Print per
column: candidate pivots, pivot chosen, any row swap, the multiplier used for
each row, and the augmented matrix after that column is cleared.
**(b)** Compute `det(A)` from the **same** elimination — no second pass —
using `det(A) = (-1)^(swaps) × prod(diag(U))`.
**(c)** Verify: report `||Ax-b||₂` (by hand, then cross-checked), compare `x`
against `np.linalg.solve` and `det` against `np.linalg.det`.

**Written:** `A[0][0]` is 0. Why is that harmless, and what would have to be
true about column 1 for the system to actually be unsolvable?

---

## E2 — Gauss-Jordan: solve *and* invert `[~30 min]`

```
A = [[ 2, -1,  3],       b = [9, 5, 19]
     [ 4,  2, -1],
     [-2,  3,  5]]
```

**(a)** Solve `Ax = b` with Gauss-Jordan, driving `[A|b]` to `[I|x]`. Print the
matrix after each column is cleared above **and** below its pivot.
**(b)** Compute `A⁻¹` by running the same sweep on `[A|I]`.
**(c)** Recover the solution a second way as `x = A⁻¹b` (hand-written
matrix-vector product) and confirm it matches (a).
**(d)** Verify against `np.linalg.solve` and `np.linalg.inv`, reporting
`||Ax-b||₂` and `||A A⁻¹ - I||_F`.

**Written:** you now have two routes to `x`. Which should you prefer in
practice, and why?

---

## E3 — `PA = LU` when naive LU fails, with three right-hand sides `[~35 min]`

```
A  = [[1, 2, 3],      b₁ = [6, 13, 11]
      [2, 4, 7],      b₂ = [1, 0, 0]
      [3, 5, 3]]      b₃ = [2, -1, 4]
```

**(a)** Attempt naive Doolittle LU (no pivoting) and show precisely where and
why it fails on this matrix.
**(b)** Redo it with partial pivoting to get `PA = LU`. Print `P`, `L`, `U` and
verify `||PA - LU||_F` — note it is `PA`, not `A`, that equals `LU`.
**(c)** Solve `Ax = b` for **all three** right-hand sides, reusing the single
`L`, `U`. Remember: solve `Lz = Pb`, never `Lz = b`.
**(d)** Compute `det(A)` from the factorization.
**(e)** Verify all three solutions against `np.linalg.solve` and report each
residual.

**Written:** naive Doolittle failed even though `A` is non-singular. Explain
the difference between "has no LU factorization" and "has no solution", and
state what `P` is doing.

---

## E4 — Largest and smallest eigenvalue → the condition number `[~30 min]`

```
A = [[9, 2, 1, 0],
     [2, 7, 1, 1],
     [1, 1, 4, 1],
     [0, 1, 1, 2]]
```

**(a)** Find the dominant eigenvalue and eigenvector by hand-written power
iteration, printing every iteration's estimate.
**(b)** Find the smallest-magnitude eigenvalue and eigenvector by hand-written
inverse power iteration (`np.linalg.inv()` allowed once for `A⁻¹`; the loop
must be yours).
**(c)** `A` is symmetric, so `κ₂(A) = |λmax| / |λmin|`. Compute it from **your**
two eigenvalues and cross-check against `np.linalg.cond(A, 2)`.
**(d)** Verify both eigenpairs against `np.linalg.eig`, comparing at unit
length and accounting for a possible sign flip.

**Written:** what does the condition number tell you about solving `Ax = b`,
and why does that make it relevant to P2's round-off question?

---

## E5 — Deflating twice: the third eigenvalue `[~35 min]`

```
A = [[6, 2, 1, 0],
     [2, 5, 1, 1],
     [1, 1, 4, 1],
     [0, 1, 1, 3]]
```

**(a)** Find `(λ₁, v₁)` by hand-written power iteration.
**(b)** Deflate: `A₂ = A - λ₁v₁v₁ᵀ`. Find `(λ₂, v₂)` from `A₂`.
**(c)** Deflate again: `A₃ = A₂ - λ₂v₂v₂ᵀ`. Find `(λ₃, v₃)` from `A₃`.
**(d)** Recover `λ₄` for free from the trace identity, with no iteration.
**(e)** Verify all four against `np.linalg.eig` and report how the error in
each successive eigenvalue grows.

**Written:** accuracy degrades with each successive deflation. Explain the
mechanism, and say what it implies about using deflation to find *all*
eigenvalues of a large matrix.

---

## E6 — `A = VΛV⁻¹`: reconstruct, power, and solve with it `[~30 min]`

```
A = [[4, 1, 1],       b = [6, 4, 3]
     [1, 3, 0],
     [1, 0, 2]]
```

**(a)** Get all eigenpairs (`np.linalg.eig` allowed — there is no short
hand-written general eigensolver). Build `V` with eigenvectors as **columns**
and `Λ = diag(λ)`, column `j` of `V` paired with `Λ[j][j]`.
**(b)** Verify `A = VΛV⁻¹` by hand-written matrix multiplication, reporting
`||A - VΛV⁻¹||_F`.
**(c)** Compute `A⁶` via `VΛ⁶V⁻¹` and via `np.linalg.matrix_power`; confirm.
**(d)** Solve `Ax = b` **through** the decomposition: `x = VΛ⁻¹V⁻¹b`. Compare
against `np.linalg.solve` and report `||Ax-b||₂`.
**(e)** Also compute `A⁻¹` as `VΛ⁻¹V⁻¹` and `A^½` as `VΛ^½V⁻¹`, confirming
`(A^½)² = A`.

**Written:** solving `Ax = b` via the eigendecomposition works, but nobody does
it in practice. Why not — and when *is* the decomposition the right tool?

---

## E7 — Classifying systems, including two free variables `[~35 min]`

```
System 1:  A = [[1,2,-1,3],[2,4,1,0],[-1,1,2,1],[3,1,0,2]]   b = [8,11,3,7]
System 2:  A = [[1,2,3,4],[2,4,6,8],[1,1,1,1],[3,3,3,3]]     b = [10,20,4,12]
System 3:  A = [[1,2,3,4],[2,4,6,8],[1,1,1,1],[3,3,3,3]]     b = [10,21,4,12]
```

For **each** system:

**(a)** Run Gaussian elimination with partial pivoting. Print the pivot chosen
and any swap per column, and the augmented matrix after each.
**(b)** Classify from the reduced form: a row `0 0 0 0 | c` with `c ≠ 0` → no
solution (print the offending equation); a row `0 0 0 0 | 0` → infinitely many
(one free variable per missing pivot); otherwise unique.
**(c)** Act on it — unique: back-substitute, verify against `np.linalg.solve`,
report `||Ax-b||₂`. Infinite: identify *which* variables are free, choose
values, produce one particular solution, and verify it against the original
system. No solution: report the contradiction, do not attempt to solve.
**(d)** Cross-check every verdict with the rank test.

**Written:** System 2 has **two** free variables. How do you know the number of
free variables before choosing any of them, and how many solutions does that
give?
