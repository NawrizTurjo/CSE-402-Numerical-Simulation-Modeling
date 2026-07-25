# Power Method

Library: [`algorithms/power_method.py`](../algorithms/power_method.py)
Exam template: [`templates/04_power_method.py`](../templates/04_power_method.py)

## Algorithm level

**Goal:** find the dominant (largest-magnitude) eigenvalue and eigenvector
of `A` without expanding a determinant.

**Why it works.** Any starting vector `x0` is (implicitly) a combination of
`A`'s eigenvectors: `x0 = c1*v1 + c2*v2 + ... + cn*vn`. Multiplying by `A`
repeatedly:
```
A^k x0 = c1*lam1^k*v1 + c2*lam2^k*v2 + ... + cn*lamn^k*vn
```
Factor out `lam1^k` (the dominant eigenvalue):
```
A^k x0 = lam1^k [ c1*v1 + c2*(lam2/lam1)^k*v2 + ... ]
```
If `|lam1| > |lam_i|` for every other `i`, each ratio `(lam_i/lam1)^k -> 0`
as `k -> infinity`. So `A^k x0` converges in *direction* to `v1`, and the
scale factor converges to `lam1`.

**Steps:**
1. Start with any nonzero `x0`.
2. `y_{k+1} = A x_k`.
3. Eigenvalue estimate = entry of `y_{k+1}` with the **largest magnitude**
   (keep its sign). Divide `y_{k+1}` by that entry to get `x_{k+1}`
   (normalization — keeps the numbers from exploding as `lam1^k -> infinity`,
   and is *why* the normalizing factor converges to `lam1`: once `x_k` is
   close to `v1` with its largest entry pinned at 1, `A x_k ≈ lam1 x_k`, so
   the new largest entry is `≈ lam1`).
4. Repeat until consecutive eigenvalue estimates stop changing.

## Code level

- `y[np.argmax(np.abs(y))]` — the entry of **largest magnitude, sign
  preserved**. This is NOT the same as `np.max(y)`, which picks the
  largest *signed* value and silently gives the wrong answer for a vector
  like `[-5, 2]` (dominant entry is -5, but `np.max` returns 2). This bug
  appears in some of the raw collected solutions in
  `res/Src-14/LU/q4_inverse_power_no_explicit_inverse.py`'s siblings —
  always use the `argmax(abs(...))` pattern.
- Two *different* normalization conventions are in play and must not be
  confused: the iteration itself normalizes by "largest entry = 1"
  (`x = y / lam_est`), but NumPy's `eig` returns eigenvectors normalized to
  unit `||.||_2`. Convert with `x = x / np.linalg.norm(x)` **once, right
  before returning/comparing** — not during the loop (that would break the
  "largest entry = 1" stability property the iteration relies on).
- Eigenvector **sign is ambiguous** (`v` and `-v` are equally valid) — flip
  NumPy's sign to match before comparing: `if np.dot(v, v_np) < 0: v_np = -v_np`.
- `is_oscillating(history)` is a diagnostic helper, not part of the core
  algorithm — plain power iteration assumes a unique dominant eigenvalue.
  If the true dominant eigenvalue is complex (comes in a conjugate pair) or
  tied with another real eigenvalue of opposite sign, the iterates
  oscillate and never settle — see `complex_cases/complex_conjugate_eigenvalues.py`
  and `complex_cases/equal_magnitude_eigenvalue_tie.py`.

## Common written questions

- **Why does the normalizing factor converge to the dominant eigenvalue?**
  See the algorithm-level derivation above (`A^k x0 -> lam1^k * c1 * v1`).
- **What breaks the method?** No single dominant eigenvalue (tie in
  magnitude, or complex-conjugate dominant pair), or `x0` with zero
  component along `v1` (blind spot — rare with a generic `x0`, common if
  `x0` happens to be symmetric with respect to some symmetry of `A`).
