# Exam cheatsheet — NumPy patterns + theory quick answers

## NumPy patterns

```python
A = np.array([[2, 1], [1, 2]], dtype=float)   # force float from the start
A = A.astype(float).copy()                     # never mutate caller's array

A[i]        # row i          A[:, j]     # column j
A[i, k:]    # row i, col k..end          A[k:, j]  # col j, row k..end
A[i, :i]    # forward-sub slice          A[i, i+1:] # backward-sub slice

A[i, k:] -= m * A[k, k:]        # vectorized elimination row-update
A[[k, p]] = A[[p, k]]           # vectorized row swap (fancy indexing)
p = k + np.argmax(np.abs(A[k:, k]))   # vectorized pivot search

A @ B        # matmul (NOT A * B -- that's elementwise!)
A[i, i+1:] @ x[i+1:]            # dot-product replacing an inner loop

np.max(np.abs(y))               # magnitude-based max
np.argmax(np.abs(y))            # INDEX of magnitude-based max (keep this for sign!)
y[np.argmax(np.abs(y))]         # value at that index, SIGN PRESERVED -- use this,
                                 # never np.max(y) alone (picks signed max, wrong
                                 # for vectors like [-5, 2])

np.linalg.solve(A, b)   np.linalg.inv(A)   np.linalg.det(A)
np.linalg.eig(A)        # (eigenvalues, eigenvectors) -- eigvecs are COLUMNS
np.linalg.norm(x)       # ||x||_2          np.linalg.norm(A, 'fro')  # Frobenius

x = x / np.linalg.norm(x)        # unit-length (NumPy's eigenvector convention)
x = x / np.max(np.abs(x))        # largest entry = 1 (power-iteration convention)

if np.dot(v, v_np) < 0: v_np = -v_np   # eigenvector sign ambiguity fix
```

**Silent traps:**
- Missing `return` in a substitution function → `None` assigned into a float array becomes `NaN`, no error.
- `A * B` where you meant `A @ B` → silently wrong if shapes broadcast.
- Integer input array → force `.astype(float)` before any elimination math, or ops may silently truncate.
- `np.max(y)` vs `y[np.argmax(np.abs(y))]` → different things; magnitude-based selection needs the second.

---

## The marking rule: hand-code your own answer, verify with NumPy after

From `res/A2_Prep.md` §1, confirmed against the real rubric. NumPy may only
**re-derive** a number you already computed yourself. These four look harmless
and are not:

| you wrote | why it is not "verification" | hand-written version |
|---|---|---|
| `np.linalg.norm(x)` to normalize your eigenvector | the norm is part of your eigenvector answer | `sum(v*v for v in x) ** 0.5` |
| `A @ x` inside your residual | `Ax` is part of your residual answer | row-by-row dot-product loop |
| `np.linalg.norm(A - L@U)` | `‖A-LU‖` *is* the answer to "verify the LU" | manual `matmul` then flatten + L2 |
| `np.outer(v, v)` in deflation | the deflated matrix is your answer | `[[u[i]*v[j] ...] ...]` |

Write it like this, every time:

```python
manual = sum(value * value for value in residual) ** 0.5   # the ANSWER
check  = np.linalg.norm(A @ x - b)                          # the CHECK
print(f"manual ||Ax-b|| = {manual:.3e}   numpy = {check:.3e}")
```

Ready-made: `algorithms/manual_ops.py` (`l2_norm`, `frobenius_norm`, `matvec`,
`matmul`, `outer`, `residual_norm`, `reconstruction_error`, `normalize`,
`normalize_max`) and `algorithms/verification.py` (`verify_solution`,
`verify_lu`, `verify_inverse`, `verify_determinant`, `verify_eigenpair`,
`verify_eigendecomposition`, `verify_matrix_power`, `verify_classification`).

The one documented exception: A1's question explicitly *named*
`np.linalg.inv()` as an allowed building block. If the question names a library
call, use it — otherwise assume it is not allowed for your own result.

---

## Gauss elimination

- Forward-eliminate → upper triangular → back-substitute.
- **Partial pivoting**: at step `k`, swap in the row (≥k) with largest `|entry|` in column `k`. Fixes both division-by-zero and round-off blowup (small pivot → huge multiplier → catastrophic cancellation). Guarantees every multiplier `|m| ≤ 1`.
- **Classify Ax=b** after elimination: no zero row → unique. Zero row `0=c, c≠0` → no solution. Zero row `0=0` → infinite solutions (pick free variable, back-solve rest).
- **Zero diagonal entry ≠ automatically "no solution"**: if some row below still has a nonzero entry in that column, pivoting fixes it and a unique solution is still possible. Only when the entire column (at & below) is zero does singularity kick in — and then the RHS decides no-solution vs infinite.
- **Determinant for free**: row ops that add/subtract multiples of rows don't change det; triangular → det = product of diagonal; each row **swap** flips the sign. `det(A) = (-1)^swaps * prod(diag(U))`.

## Gauss-Jordan

- Same as Gauss elimination but: normalize each pivot row to 1, and eliminate the pivot variable from **every** other row (above *and* below) → augmented matrix drives straight to `[I | x]`. No back substitution. Same pitfalls/fixes (pivoting) as Gauss elimination, just more arithmetic.
- **Matrix inverse — same loop, wider right-hand block**: `[A | I]` → `[I | A⁻¹]`. Works because the sweep applies row operations `E_k…E_1` with `(E_k…E_1)A = I`, so `E_k…E_1` *is* `A⁻¹`, and those same operations turn the right-hand `I` into it. Equivalently: column `j` of the answer solves `Ax_j = e_j` — the identity's columns are just `n` right-hand sides. Remember to sweep all `2n` columns when normalizing/eliminating.
- **Determinant needs the RAW pivots**: `det(A) = (-1)^swaps × prod(pivot before its row was normalized)`. Normalizing a row by `p` divides `det` by `p`, so by full RREF every pivot is 1 and `prod(diag) = 1` regardless of the true determinant — the information is destroyed, not hidden. Record each pivot the instant it is found. (Plain Gauss elimination never normalizes, so `prod(diag(U))` at the end is fine there.)
- **Solve directly, don't invert-then-multiply.** `x = A⁻¹b` is how the maths is *written*; `solve A x = b` is how it should be *computed* — inverting costs ~n× more arithmetic and is numerically worse, since every entry of `A⁻¹` carries its own round-off which then mixes into every component of `x`.

## LU decomposition

- `A = LU` (Doolittle: `L` has 1s on diagonal). `U` is exactly what Gauss forward elimination produces; `L` stores the multipliers `m = U[i,k]/U[k,k]` at position `(i,k)` — this falls out of elimination = product of elementary matrices collapsing to a single unit-lower-triangular matrix.
- **Existence (no pivoting)**: iff every **leading principal minor** is nonzero. `det(A)≠0` alone is NOT enough. Counterexample: `[[0,1],[1,0]]` (det=-1, but leading 1×1 minor = 0).
- **`PA = LU`** (partial pivoting) exists for **every invertible matrix**.
- Solve: `Lz = P@b` (NOT `Lz = b`!), then `Ux = z`.
- Row swap mid-elimination must also swap **already-computed multipliers** in L's earlier columns: `L[[k,p], :k] = L[[p,k], :k]`.
- **Cost**: factor once `O(n³)` (~2n³/3 flops); each solve (given factors) `O(n²)`. `k` different b's: LU = `O(n³ + k·n²)` vs. repeated Gauss = `O(k·n³)` — LU wins once `k>~1`.
- **Determinant**: `det(A) = (-1)^swaps * prod(diag(U))`.
- **Inverse**: column `j` of `A⁻¹` solves `A x_j = e_j` — factor once, solve `n` times. **Never** form `A⁻¹` just to solve `Ax=b` (2n² per solve vs. 2n³ total to build the inverse first — n× more expensive, and compounds rounding error).
- **Cholesky** (`A = LLᵀ`) replaces LU when `A` is symmetric positive-definite: ~half the cost (`n³/3`), no pivoting needed (SPD guarantees positive pivots).
- **Small residual ≠ accurate answer**: check condition number `κ(A) = ‖A‖·‖A⁻¹‖`. `relative error in x ≲ κ(A) × relative residual`.

## Power method

- `x_{k+1} = normalize(A x_k)`; normalizing factor (entry of **largest magnitude**, sign kept) → dominant eigenvalue estimate. Converges because `Aᵏx₀ ≈ λ₁ᵏc₁v₁` once `|λ₁| > |λᵢ|` for all others.
- Normalize by `y[argmax(|y|)]` during iteration (keeps numbers bounded); normalize to unit `‖·‖₂` **once at the end** before comparing to `np.linalg.eig`.
- Blind spot: if `x0` has zero component along the true dominant eigenvector (common with symmetric starting guesses like `[1,1,...,1]` after deflation), convergence jumps to the wrong eigenvalue. Use a generic, non-symmetric `x0`.

## Inverse power method (smallest eigenvalue)

- `λ` eigenvalue of `A` ⟺ `1/λ` eigenvalue of `A⁻¹`. Run power method on `A⁻¹` → converges to `1/λ_min` → reciprocate.
- **Never form `A⁻¹` explicitly per iteration** — factor `A=LU` once, then each iteration = two triangular solves (`Lz=x`, `Uy=z`). Cheaper setup (`O(n³)` once vs `O(n³)` to invert anyway, but sparsity is preserved in L,U — `A⁻¹` of a sparse matrix is generally dense).

## Deflation (intermediate eigenvalues)

- Power method only ever finds the dominant eigenvalue. To get `λ₂`: normalize `v₁` to unit length, deflate `A₂ = A − λ₁ v̂₁v̂₁ᵀ`, run power method on `A₂`.
- Works cleanly for **symmetric** `A` (orthogonal eigenvectors → `v̂₁ᵀv₂ = 0` → `v₂` untouched by deflation).
- Errors compound with each deflation step — reliable for first few eigenvalues only.
- Same blind-spot trap as plain power method applies at every deflation stage.

## Eigenvalue decomposition `A = VΛV⁻¹`

- `V` columns = eigenvectors, `Λ` = diagonal of matching eigenvalues (same order).
- Derivation: `Avᵢ=λᵢvᵢ` for every `i` → `AV = VΛ` → `A = VΛV⁻¹`.
- Useful for `Aᵏ = VΛᵏV⁻¹` — the inner `V⁻¹V` pairs telescope away for every intermediate power; only translate once at each end, and `Λᵏ` is trivial (diagonal).
- **Any function of `A`, same trick**: `A⁻¹ = VΛ⁻¹V⁻¹` (reciprocate the eigenvalues), `A^½ = VΛ^½V⁻¹` (square-root them). Once diagonalized, `f(A)` is `f` applied to `n` scalars.
- Needs `A` **diagonalizable** — `n` independent eigenvectors, or `V` is singular. Symmetric ⇒ always fine (spectral theorem). Defective example: `[[2,1],[0,2]]` (repeated `λ=2`, only one eigenvector).
- Eigenvector sign & ordering are arbitrary — NumPy returns them in no particular order and with an arbitrary sign per vector. Build `V` with `np.column_stack((v1, v2))`, **not** `np.array([v1, v2])` (that gives `Vᵀ`).

## Characteristic polynomial (direct, non-iterative)

- **2×2, memorize this**: `λ² − tr(A)·λ + det(A) = 0`, so `λ = [tr ± √(tr²−4det)] / 2`. Discriminant `> 0` two distinct real, `= 0` repeated, `< 0` complex-conjugate pair (⇒ `|λ₂/λ₁| = 1` ⇒ the power method will never converge on it).
- **3×3**: `λ³ − tr(A)λ² + (sum of the three 2×2 principal minors)λ − det(A) = 0`, then root it.
- **Eigenvector for a known `λ`**: `A − λI` is singular by construction; row-reduce it, set the free variable to 1, back-substitute, normalize. For a 2×2 one row is enough: `B₀₀v₁ + B₀₁v₂ = 0` ⇒ `v = (1, −B₀₀/B₀₁)`.
- **Two free checks on ANY eigenvalue answer, any `n`, any method**: `Σλᵢ = tr(A)` and `Πλᵢ = det(A)`. Two lines that catch a dropped, duplicated, or mis-signed eigenvalue instantly.

## Condition number

- `κ(A) = ‖A‖·‖A⁻¹‖`; for **symmetric** `A`, `κ₂ = |λmax| / |λmin|` — so the power method and inverse power method together give it to you.
- Meaning: `‖δx‖/‖x‖ ≤ κ · ‖δb‖/‖b‖`. You lose roughly `log₁₀(κ)` decimal digits. `np.linalg.cond(A, 2)` to check.
- **Distinct from the pivoting/round-off failure.** Bad pivoting is an *algorithm* problem (fixable — use partial pivoting). Ill-conditioning is *inherent* to the matrix (no algorithm helps).
- Consequence for verification: a small `‖Ax−b‖` does **not** prove `x` is accurate. On an ill-conditioned `A` the residual can be `1e-15` with `x` wrong in the second decimal. Print `np.linalg.cond(A)` alongside the residual whenever the matrix looks suspicious.
