# Eigenvalue Decomposition (A = V Λ V⁻¹)

Library: [`algorithms/eigen_decomposition.py`](../algorithms/eigen_decomposition.py)
Exam template: [`templates/07_eigenvalue_decomposition.py`](../templates/07_eigenvalue_decomposition.py)

## Algorithm level

**Setup:** if `A` has `n` linearly independent eigenvectors
`v1, ..., vn` with eigenvalues `lam1, ..., lamn`, pack the eigenvectors as
**columns** of a matrix `V = [v1 v2 ... vn]` and the eigenvalues on the
diagonal of `Lambda`.

**Derivation (not just stated):** `Av_i = lam_i v_i` holds for every `i`.
Write all `n` equations side by side as one matrix equation:
```
A [v1 v2 ... vn] = [lam1*v1  lam2*v2  ...  lamn*vn]
```
The left side is `AV`. The right side factors as `V @ diag(lam1,...,lamn)`
(each column of `V` scaled by its own eigenvalue is exactly what
multiplying by a diagonal matrix on the right does). So:
```
AV = V Lambda   =>   A = V Lambda V^-1
```

**Reading `Ax = V Lambda V^-1 x` right to left, as a story:**
1. `V^-1 x` — rewrite `x` in eigenvector coordinates ("how much of `x`
   points along each `v_i`?").
2. `Lambda (V^-1 x)` — do what `A` actually does: scale each direction
   independently by its own eigenvalue. This is the only step that does
   real work; `V` and `V^-1` are just translators between coordinate
   systems.
3. `V (...)` — translate the scaled result back to standard coordinates.

**Why this is useful — `A^k`:** `A^2 = (VΛV^-1)(VΛV^-1) = VΛ(V^-1V)ΛV^-1 =
VΛ^2V^-1` — the middle `V^-1V` pair is the identity and cancels. By
induction, `A^k = V Λ^k V^-1`. Since `Λ` is diagonal, `Λ^k` just raises
each diagonal entry to the `k`-th power — trivial. You translate **once**
at the start and **once** at the end; all `k` "multiplications" happen for
free in the easy (diagonal) coordinate system.

## Code level

- `V = evecs` directly from `np.linalg.eig(A)` — NumPy already returns
  eigenvectors as **columns**, matching the `V` convention exactly; no
  transpose needed.
- `Lambda = np.diag(evals)` — same order as `V`'s columns, since `evals`
  and `evecs` come from the same `np.linalg.eig` call and are already
  paired correctly.
- `inverse_2x2` is a hand-written closed-form fallback
  (`[[d,-b],[-c,a]]/det`) for when a question specifically bans
  `np.linalg.inv()` — only valid for exactly 2x2 input.
- Complex matrices: `np.linalg.eig` can return complex `evals`/`evecs`
  even for a real input matrix (e.g. a rotation-like `A`). This file's
  `verify_via_numpy` takes `.real` when reconstructing — appropriate only
  if you've confirmed the imaginary parts are genuinely ~0 (a real
  diagonalizable matrix with real eigenvalues). If `A` actually has complex
  eigenvalues, discarding the imaginary part silently produces a WRONG
  reconstruction — see `complex_cases/complex_conjugate_eigenvalues.py`
  for what real complex eigenvalues look like and how to verify the
  decomposition correctly in that case (don't drop `.real`).

## Common written questions

- **Why is `Lambda` diagonal?** Each eigenvector direction is scaled
  *independently* by its own eigenvalue — no mixing between directions —
  and that independence is exactly what a diagonal matrix encodes.
- **Derivation in one line?** `Av_i=lam_i*v_i` for every `i` ⟹
  `A[v1...vn] = [v1...vn]diag(lam1..lamn)` ⟹ `AV=VΛ` ⟹ `A=VΛV^-1`.
- **What if `A` isn't diagonalizable (repeated eigenvalue with too few
  independent eigenvectors)?** `V` becomes singular (can't invert) — this
  decomposition doesn't exist as-is; the generalization is the Jordan
  normal form (out of scope for this syllabus, but worth naming if asked
  "does this always work?").
