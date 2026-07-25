# Hotelling's Deflation

Library: [`algorithms/deflation.py`](../algorithms/deflation.py)
Exam template: [`templates/06_deflation_intermediate_eigen.py`](../templates/06_deflation_intermediate_eigen.py)

## Algorithm level

**Goal:** find the **second**, **third**, ... eigenvalues after the power
method has already found the dominant one — the power method by itself
only ever finds the loudest eigenvalue.

**Idea:** "mute" the dominant eigenvalue's contribution and re-run power
iteration on what's left.

Any vector `x` decomposes as `x = sum_i c_i * v_i`. `A`'s action on `x` is
`Ax = sum_i c_i * lam_i * v_i` — each eigen-direction scaled independently.
The **v1-contribution** to `Ax` specifically is `lam1 * (v1-component of x)`.
The v1-component of any `x` can be extracted by projection onto the unit
eigenvector: `v1_hat * (v1_hat . x)`. So the v1-contribution to `Ax` is
`lam1 * v1_hat * (v1_hat . x)` — subtract it:
```
A2 x = Ax - lam1 * v1_hat * (v1_hat . x) = (A - lam1 * v1_hat @ v1_hat.T) x
=> A2 = A - lam1 * v1_hat @ v1_hat.T
```

**Why the other eigenvectors survive untouched (for symmetric `A`):**
eigenvectors of a *symmetric* matrix are orthogonal, so
`v1_hat . v2 = 0` — the projection that would remove v1's contribution
finds nothing to subtract when applied to `v2`:
```
A2 v2 = A v2 - lam1 * v1_hat * (v1_hat . v2) = lam2 * v2 - 0 = lam2 * v2
```
`v2` is still an eigenvector of `A2` with the **same** eigenvalue `lam2` —
now dominant in `A2` since `lam1`'s direction was zeroed out.

**Steps:**
1. Power method on `A` → `lambda1, v1`.
2. Normalize: `v1_hat = v1 / sqrt(v1^T v1)`.
3. Deflate: `A2 = A - lambda1 * v1_hat @ v1_hat.T`.
4. Power method on `A2` → `lambda2, v2`. Repeat (`A3 = A2 - lambda2*v2_hat@v2_hat.T`, ...) for further eigenvalues.

**Caveat (from the theory slides directly):** each deflation step uses an
*approximate* eigenvector (power iteration only converges in the limit).
Errors compound with every deflation — reliable for the first few
eigenvalues, degrades if you try to peel off too many.

## Code level

- `deflate(A, lam, v)` is a two-line function but relies entirely on `A`
  being symmetric for the "leaves other eigenvectors untouched" guarantee
  — don't reach for it on a non-symmetric matrix without checking
  `docs/deflation.md`'s written-question note below first.
- `find_all_eigenpairs` loops `power_iteration -> deflate` `n` times,
  reusing the SAME `x0` at every stage. **This is the file's biggest
  landmine**: `x0` must not be a blind spot for *any* of the deflated
  matrices along the way, not just the original `A`. A concrete example
  that broke: `x0=[1,1,1,1]` finds `lambda1=10` fine, but the deflated
  matrix's next dominant eigenvector is `[1,-1,0,0]`-direction — and
  `[1,1,1,1]` has **zero component** along it. Power iteration then jumps
  straight to the *third* eigenvalue with no error or warning at all. Use
  a generic, non-symmetric `x0` (e.g. `[1,2,3,-1]`) — verified in this
  file's `_self_test()` to recover all four eigenvalues of the standard
  block-diagonal example matrix (10, 6, 4, 2) correctly.
- `_power_iteration` here is a trimmed private copy of
  `algorithms/power_method.py`'s function, kept local for the same
  copy-paste self-containment reason as elsewhere.

## Common written questions

- **Why does deflation leave `v2` untouched?** See the orthogonality
  argument above — `v1_hat . v2 = 0` because eigenvectors of a symmetric
  matrix are orthogonal.
- **Does this work for non-symmetric matrices?** Not cleanly with this
  formula — non-symmetric eigenvectors aren't orthogonal in general, so
  subtracting `lam1 * v1_hat @ v1_hat.T` can also perturb the *other*
  eigenvectors' directions, not just mute `v1`.
- **Power method on the deflated matrix "converges" suspiciously fast to a
  smaller-than-expected eigenvalue — what happened?** Blind spot: `x0` has
  ~zero component along the new dominant eigenvector. Switch to a generic
  `x0` and re-run.
