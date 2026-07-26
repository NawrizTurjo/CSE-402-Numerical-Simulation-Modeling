# Exam code — index

Everything here is tested and runnable as-is (`py -3 <file>.py`, exit 0,
cross-checked against `np.linalg` and the slide-worked examples).

```
code/
├── algorithms/       canonical library functions -- one file per topic, the
│                      source of truth every doc/template/complex-case builds on
├── docs/              one .md per algorithm: algorithm-level (math/derivation)
│                      vs. code-level (why the implementation looks the way it does)
├── templates/         exam-ready standalone scripts (grab the whole file,
│                      swap in the question's matrix, run)
├── practice/          13 timed mock questions with full solutions, plus a
│                      statements-only questions.md for blind practice
├── complex_cases/      harder variants: rank deficiency >1, singular matrices,
│                      ill-conditioning, complex/tied eigenvalues, larger n
├── prev-solutions/    solutions mapped to the 5 REAL transcribed past
│                      questions (res/Online Questions.txt) + a slides GJ ref
└── cheatsheet.md       one-page NumPy syntax + theory quick-answers
```

## Which folder do I actually open?

1. **Know the topic, need the algorithm right now** → `algorithms/<topic>.py`.
   Minimal, clean, copy-paste a single function straight into your answer.
2. **Want the full worked pattern** (prints, NumPy verification, plots) for
   a question shaped like "solve this system / find this eigenpair" →
   `templates/<NN>_<topic>.py`. This is what most of your exam answer will
   look like.
3. **Something looks wrong / too easy / won't converge** → check
   `complex_cases/README.md` first — oscillating power method, `nan`-filled
   LU, and "small residual but wrong answer" are all diagnosed there with
   the fix.
4. **Want to see exactly what a real graded question needed** →
   `prev-solutions/README.md`.
5. **Practising against the clock** → `practice/questions.md` (statements
   only), solutions in `practice/a2prep/` and `practice/extra/`.
6. **Need the "why", not just the "how"** → `docs/<topic>.md` (paired
   1:1 with `algorithms/`).
7. **5-second syntax/theory lookup** → `cheatsheet.md`.

| Topic | Library | Docs | Template |
|---|---|---|---|
| Gauss elimination | [algorithms/gauss_elimination.py](algorithms/gauss_elimination.py) | [docs/gauss_elimination.md](docs/gauss_elimination.md) | [templates/01_gauss_elimination.py](templates/01_gauss_elimination.py) |
| Gauss-Jordan (+ inverse, RREF, det) | [algorithms/gauss_jordan.py](algorithms/gauss_jordan.py) | [docs/gauss_jordan.md](docs/gauss_jordan.md) | [templates/02_gauss_jordan.py](templates/02_gauss_jordan.py) |
| LU decomposition | [algorithms/lu_decomposition.py](algorithms/lu_decomposition.py) | [docs/lu_decomposition.md](docs/lu_decomposition.md) | [templates/03_lu_decomposition.py](templates/03_lu_decomposition.py) |
| Power method | [algorithms/power_method.py](algorithms/power_method.py) | [docs/power_method.md](docs/power_method.md) | [templates/04_power_method.py](templates/04_power_method.py) |
| Inverse power method | [algorithms/inverse_power_method.py](algorithms/inverse_power_method.py) | [docs/inverse_power_method.md](docs/inverse_power_method.md) | [templates/05_inverse_power_method.py](templates/05_inverse_power_method.py) |
| Deflation | [algorithms/deflation.py](algorithms/deflation.py) | [docs/deflation.md](docs/deflation.md) | [templates/06_deflation_intermediate_eigen.py](templates/06_deflation_intermediate_eigen.py) |
| Eigenvalue decomposition | [algorithms/eigen_decomposition.py](algorithms/eigen_decomposition.py) | [docs/eigen_decomposition.md](docs/eigen_decomposition.md) | [templates/07_eigenvalue_decomposition.py](templates/07_eigenvalue_decomposition.py) |
| Characteristic polynomial | [algorithms/characteristic_polynomial.py](algorithms/characteristic_polynomial.py) | — | [practice/a2prep/p3_characteristic_polynomial.py](practice/a2prep/p3_characteristic_polynomial.py) |
| Hand-written norms/products | [algorithms/manual_ops.py](algorithms/manual_ops.py) | — | — |
| Verify-against-NumPy helpers | [algorithms/verification.py](algorithms/verification.py) | — | — |
| Curve fitting (bonus) | — | — | [templates/08_curve_fitting_bonus.py](templates/08_curve_fitting_bonus.py) |
| Plotting | — | — | [templates/plotting_template.py](templates/plotting_template.py) |

Every `algorithms/*.py` file ends with a `_self_test()` that runs on
`py -3 algorithms/<file>.py` — if you edit one of these before the exam,
re-run its self-test.

## Two rules that decide marks, not just style

Both are recorded in `res/A2_Prep.md` section 1, one of them confirmed against
the real marking scheme.

**1. Hand-code anything that is part of YOUR answer.** NumPy is for
cross-checking afterwards, not for producing pieces of the result. The
non-obvious cases — these look too small to matter and are the easiest marks
to drop:

| looks harmless | why it is not "verification" | use instead |
|---|---|---|
| `np.linalg.norm(x)` to normalize your eigenvector | the norm is part of your eigenvector answer | `manual_ops.l2_norm` |
| `A @ x` when computing a residual | `Ax` is part of your residual answer | `manual_ops.matvec` |
| `np.linalg.norm(A - L@U)` | `‖A-LU‖` *is* the answer to "verify the LU" | `manual_ops.reconstruction_error` |
| `np.outer(v, v)` in deflation | the deflated matrix is your answer | `manual_ops.outer` |

Pattern to write every time:
```python
manual_value = l2_norm(residual)                    # this is the answer
numpy_value  = np.linalg.norm(A @ x - b)            # this is the check
print(f"manual = {manual_value:.3e}   numpy = {numpy_value:.3e}")
```

**2. Verify against the library at the end, every time.** Every algorithm
module now exposes a `verify_*` function that does the whole comparison in one
call, and `algorithms/verification.py` has generic versions:

```python
verify_solution(A, b, x)              verify_eigenpair(A, lam, v)
verify_lu(A, L, U, P=P)               verify_eigenpairs(A, lams, vecs)
verify_inverse(A, A_inv)              verify_eigendecomposition(A, V, Lam)
verify_determinant(A, det_value)      verify_matrix_power(A, k, A_k)
verify_classification(A, b, "unique")
```

Each hand-computes the error first, prints the NumPy value beside it, and
returns `True`/`False` so it also works inside an `assert`.

## Before the exam

1. Skim `cheatsheet.md` once.
2. Skim `complex_cases/README.md` once — just the symptom table, so you
   recognize a "hard case" instead of debugging blind if you hit one.
3. Confirm your exam machine has `numpy` (and `matplotlib` if plots are
   asked):
   ```
   py -3 -c "import numpy, matplotlib; print('ok')"
   ```
   (On this machine, plain `python`/`python3` hit the Windows Store alias —
   use `py -3` instead. Check what launcher the exam machine actually has.)

4. Do at least two timed mocks from `practice/questions.md` — statements
   only, 30-minute timer, blank editor. Reading builds recognition; only
   timed production builds the speed the exam actually measures.

## During the exam

1. Check `prev-solutions/README.md` first in case the question is close to
   one of the 5 real transcribed ones.
2. Otherwise, open the matching `templates/<NN>_<topic>.py`, copy the
   relevant function(s) into your answer file (or edit the `__main__`
   block in place if the exam allows submitting a full script).
3. Replace the example `A`, `b`, `x0` with the values given in the question.
4. Keep hand-written pivot/swap loops (the default in every file) unless
   the question explicitly allows `np.argmax`/fancy indexing for that part.
5. Every solver prints intermediate state (pivots, swaps, L/U after each
   column, augmented matrix after each cleared column) — past questions
   have explicitly asked for this. Leave the prints in.
6. Always keep the `np.linalg.*` verification lines already present in
   each `__main__` — "verify with NumPy" + a residual/error norm has been
   part of nearly every past question.

## Known gotchas already fixed in this code (don't reintroduce them)

- Eigenvalue estimate must come from `y[np.argmax(np.abs(y))]`, not
  `np.max(y)` (wrong sign/value for vectors with a dominant negative entry).
- Deflation / power method starting vector `x0` must not be blind to the
  target eigenvector (symmetric guesses like `[1,1,1,1]` can have zero
  component along the eigenvector you're trying to find after deflation —
  see `docs/deflation.md`).
- When solving `PA=LU` systems, solve `Lz = P@b`, never `Lz = b`.
- Row swaps mid-LU-factorization must also swap `L`'s already-stored
  multipliers in earlier columns.
- Eigenvector sign is ambiguous — flip NumPy's sign to match yours
  (`if np.dot(v, v_np) < 0: v_np = -v_np`) before comparing/subtracting.
- Force `.astype(float).copy()` on every input matrix before doing
  arithmetic on it.
- A zero diagonal entry during elimination does NOT automatically mean "no
  solution" — see `docs/gauss_elimination.md`.
- Small `||Ax-b||` does NOT guarantee an accurate `x` — check the
  condition number (`complex_cases/ill_conditioned_systems.py`,
  `practice/extra/e4_power_and_inverse_power_condition_number.py`).
- Computing a norm/residual/`A@x` that is part of your own answer with
  `np.linalg` instead of a loop — see the marking rule above.
- Gauss-Jordan determinant: capture each pivot BEFORE normalizing the row.
  Once the matrix is in RREF every pivot is 1 and `det` is unrecoverable.
- Deflation needs the eigenvector at UNIT length, not the power method's
  largest-entry-normalized output.
