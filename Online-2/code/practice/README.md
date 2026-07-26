# Practice problems — timed mock exams

Thirteen full, exam-shaped problems. Every one is a complete runnable script
that ends in `ALL CHECKS PASSED`, prints every intermediate step a past
question has ever asked for, and answers its written follow-up in a trailing
comment block.

```
practice/
├── questions.md   problem statements ONLY — open this for a blind timed mock
├── a2prep/        the 6 self-authored questions from res/A2_Prep.md (P1a–P5),
│                   turned into runnable, executed, verified scripts
└── extra/         7 further problems written for this repo (E1–E7), filling
                    the gaps A2_Prep.md's set left open
```

## How to actually use this

**Do not read the solutions first.** Reading builds recognition; only timed
production builds the speed a 30-minute live-coding exam tests.

1. Open [questions.md](questions.md) — statements only, no code.
2. Pick one. **Set a 30-minute timer.** Close every other file.
3. Solve it cold in a blank editor. Print pivots, swaps, intermediate
   matrices, the NumPy verification, and the residual — past questions have
   explicitly asked for all of these.
4. **Stop at 30 minutes**, done or not. That is the real constraint.
5. Only then open the solution file and diff against it. Note the ONE thing
   that cost you the most time, and drill just that before the next mock.

## The problems

| # | Problem | Topics | Time | Written Q |
|---|---|---|---|---|
| P1a | [Gauss-Jordan RREF & determinant](a2prep/p1a_gauss_jordan_rref_determinant.py) | Gauss-Jordan, det from raw pivots | ~25 min | why the pivots must be captured *before* normalization |
| P1b | [Matrix inverse: GJ vs LU](a2prep/p1b_matrix_inverse_gj_and_lu.py) | inverse via `[A\|I]`, inverse via LU | ~30 min | which method for a 1000×1000 matrix |
| P2 | [Round-off error & complexity](a2prep/p2_roundoff_error_and_complexity.py) | naive vs pivoted at 3-decimal precision | ~25 min | flop counts, Gauss-per-RHS vs LU-reused |
| P3 | [Characteristic polynomial](a2prep/p3_characteristic_polynomial.py) | `det(A-λI)=0`, quadratic formula, `(A-λI)v=0` | ~25 min | — |
| P4 | [Eigendecomposition & matrix powers](a2prep/p4_eigendecomposition_matrix_powers.py) | `A=VΛV⁻¹`, eigen-coordinates, `A⁵` | ~30 min | — |
| P5 | [Deflation](a2prep/p5_deflation_intermediate_eigenvalues.py) | power method + Hotelling deflation | ~30 min | why deflation needs a symmetric `A` |
| E1 | [Gauss on a 4×4 + determinant](extra/e1_gauss_4x4_with_determinant.py) | elimination, pivoting, det byproduct | ~25 min | zero pivot vs unsolvable system |
| E2 | [Gauss-Jordan: solve AND invert](extra/e2_gauss_jordan_solve_and_inverse.py) | `[A\|b]` and `[A\|I]`, `x = A⁻¹b` | ~30 min | solve directly vs invert-then-multiply |
| E3 | [PA=LU when naive LU fails](extra/e3_plu_when_naive_lu_fails.py) | pivoted LU, `Lz = Pb`, 3 RHS reused, det | ~35 min | "no LU" vs "no solution"; what `P` does |
| E4 | [Power + inverse power → condition number](extra/e4_power_and_inverse_power_condition_number.py) | both eigen-iterations, `κ₂ = \|λmax\|/\|λmin\|` | ~30 min | what `κ` says about solving `Ax=b` |
| E5 | [Deflating twice](extra/e5_double_deflation_third_eigenvalue.py) | repeated deflation, trace shortcut, error growth | ~35 min | why accuracy degrades per deflation |
| E6 | [Eigendecomposition: solve, power, `f(A)`](extra/e6_eigendecomposition_solve_and_powers.py) | `A⁶`, `A⁻¹`, `A^½`, solving via the eigenbasis | ~30 min | why nobody solves `Ax=b` this way |
| E7 | [Classifying systems, 2 free variables](extra/e7_classify_three_systems_free_variables.py) | rank, unique/infinite/no-solution, parametrizing | ~35 min | counting free variables from the rank |

## Coverage against the real past questions

The five transcribed real questions (`res/Online Questions.txt`) are solved in
[`../prev-solutions/`](../prev-solutions/README.md), not here — those are exact
reconstructions of what was actually asked. This folder is everything *else*
the syllabus can produce.

| Syllabus topic | Real question | Practice here |
|---|---|---|
| Gauss elimination + partial pivoting | ✅ B1 | E1, E7 |
| Classifying unique/infinite/no-solution | ✅ B1 | E7 |
| Round-off error, why pivoting exists | — | P2 |
| Determinant via elimination | — | P1a, E1, E3 |
| Gauss-Jordan / RREF | ✅ B2 | P1a, E2 |
| Matrix inverse via `[A\|I]` | ✅ B2 | P1b, E2 |
| LU decomposition (Doolittle) | ✅ C2 | P1b, E3 |
| LU with pivoting (`PA=LU`) | — | E3 |
| Reusing one LU for many RHS | ✅ C2 | P1b, E3 |
| Matrix inverse via LU | — | P1b |
| Complexity / flop counts | — | P2, P1b |
| Characteristic polynomial `det(A-λI)=0` | — | P3 |
| Eigendecomposition `A=VΛV⁻¹` | — | P4, E6 |
| Matrix powers `Aᵏ` | — | P4, E6 |
| Power method | ✅ C1 | E4, E5 |
| Inverse power method | ✅ A1 | E4 |
| Deflation | — | P5, E5 |
| Condition number | — | E4 |

Every syllabus row has at least one problem. The ✅ rows are the ones with
real precedent and deserve the most drilling.

## Suggested rotation if you only have time for three mocks

1. **E7** — B1's real question shape, one difficulty step up. Classification
   under time pressure is where careless marks go.
2. **E3** — C2's real question shape, but on a matrix where naive LU actually
   fails. If the exam matrix needs pivoting and you have only drilled the
   no-pivot version, this is the run that saves you.
3. **E4** — covers both eigenvalue questions that have actually been set (C1's
   power method and A1's inverse power method) in one sitting.

Then, if there is time: **P5 or E5** for deflation — the highest-risk untested
topic, and the last one taught.

## Two rules every solution here follows

Both come from `res/A2_Prep.md` section 1 and cost real marks:

1. **Hand-code anything that is part of your answer.** Norms, residuals,
   `A @ x`, `||A - LU||`, normalizing your own eigenvector — all of these are
   *your result*, not verification. Every file computes them with explicit
   loops and prints the NumPy value beside them purely as a same-number check.
   See [`../algorithms/manual_ops.py`](../algorithms/manual_ops.py).
2. **Verify against the library at the end, and report both numbers.** Every
   file ends with `np.linalg.solve` / `inv` / `det` / `eig` comparisons and an
   error norm. See [`../algorithms/verification.py`](../algorithms/verification.py)
   for these as ready-made functions.
