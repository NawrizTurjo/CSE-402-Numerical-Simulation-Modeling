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

## Gauss elimination

- Forward-eliminate → upper triangular → back-substitute.
- **Partial pivoting**: at step `k`, swap in the row (≥k) with largest `|entry|` in column `k`. Fixes both division-by-zero and round-off blowup (small pivot → huge multiplier → catastrophic cancellation). Guarantees every multiplier `|m| ≤ 1`.
- **Classify Ax=b** after elimination: no zero row → unique. Zero row `0=c, c≠0` → no solution. Zero row `0=0` → infinite solutions (pick free variable, back-solve rest).
- **Zero diagonal entry ≠ automatically "no solution"**: if some row below still has a nonzero entry in that column, pivoting fixes it and a unique solution is still possible. Only when the entire column (at & below) is zero does singularity kick in — and then the RHS decides no-solution vs infinite.
- **Determinant for free**: row ops that add/subtract multiples of rows don't change det; triangular → det = product of diagonal; each row **swap** flips the sign. `det(A) = (-1)^swaps * prod(diag(U))`.

## Gauss-Jordan

- Same as Gauss elimination but: normalize each pivot row to 1, and eliminate the pivot variable from **every** other row (above *and* below) → augmented matrix drives straight to `[I | x]`. No back substitution. Same pitfalls/fixes (pivoting) as Gauss elimination, just more arithmetic.

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
- Eigenvector sign & ordering are arbitrary — NumPy returns them in no particular order and with an arbitrary sign per vector.
