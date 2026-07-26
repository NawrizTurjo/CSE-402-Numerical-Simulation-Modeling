# Previous solutions — mapped to real past questions

`res/Online Questions.txt` transcribes **5** real online-exam
questions from past sections (B1/A1/C2/B2).
This folder maps each one to a working, tested solution, plus a slides-based
reference for Gauss-Jordan kept alongside B2's real one for cross-checking
against the lecture's own worked numbers.

| Real question (from `res/Online Questions.txt`) | Solved here | Notes |
|---|---|---|
| **C1's Power Iteration** — 4×4, normalize the eigenvector before comparing, expect a possible sign flip | [power_method/q_dominant_eigenpair_4x4.py](power_method/q_dominant_eigenpair_4x4.py) | **Exact matrix**, recovered from `res/A2_Prep.md` §5: `[[12,2,1,0],[2,5,0,1],[1,0,3,1],[0,1,1,2]]`. Dominant eigenvalue ≈ `12.640124`, converges in ~15 iterations. |
| **B1's system-of-equations question** — 3 systems, hand-written partial pivot, classify unique/no-solution/infinite, print every pivot+swap+post-elimination matrix, verify unique case vs NumPy | [gauss_elimination/b1_three_systems_classify_pivot.py](gauss_elimination/b1_three_systems_classify_pivot.py) | Exact match to the transcribed question's requirements, including the written-question answer (zero-diagonal-entry theory) in the trailing comment. |
| **A1's inverse power method** — exact matrix `[[4,1,0,0],[1,3,1,0],[0,1,2,1],[0,0,1,1]]`, explicitly using `np.linalg.inv()`, verify vs NumPy | [inverse_power_method/a1_smallest_eigenpair_explicit_inverse.py](inverse_power_method/a1_smallest_eigenpair_explicit_inverse.py) | Matches exact matrix in prompt using `np.linalg.inv()`. Smallest eigenvalue ≈ `0.254719`. |
| **C2's LU decomposition question** — 3×3 matrix `A`, two RHS `b1, b2`, reuse L,U factorization, verify norms `||A-LU||` and `||Ax-b||_2`, verify vs `np.linalg.solve()` | [lu_decomposition/c2_lu_two_rhs_reuse_factorization.py](lu_decomposition/c2_lu_two_rhs_reuse_factorization.py) | **Exact data**, recovered from `res/A2_Prep.md` §6: `A=[[4,3,2],[2,5,3],[1,2,4]]`, `b1=[1,2,3]`, `b2=[4,5,6]`. No pivoting needed (`‖A-LU‖ = 0`); if a variant needs it, see [practice/extra/e3](../practice/extra/e3_plu_when_naive_lu_fails.py). Includes the written answer on why L, U are not recomputed. |
| **B2's Gauss-Jordan question** — augmented `[A\|b1\|b2]`, solved with one Gauss-Jordan sweep, no row swap needed; the *same* sweep re-run on `[A\|I]` gives `A⁻¹` | [gauss_jordan/b2_two_rhs_and_inverse.py](gauss_jordan/b2_two_rhs_and_inverse.py) | **Exact data**, given directly in the question: `A=[[1,1,1],[1,2,3],[1,3,6]]`, `b1=[1,2,1]`, `b2=[2,1,-1]`. Solutions `x1=[-2,5,-2]`, `x2=[2,1,-1]`; `A⁻¹=[[3,-3,1],[-3,5,-2],[1,-2,1]]`. Written answers (why augmenting works, why it isn't ~2x cost, what replaces `b1\|b2` for the inverse) are in the trailing comment. |
| *(slides reference, not one of the 5 transcribed questions)* Gauss-Jordan | [gauss_jordan/slides_worked_example.py](gauss_jordan/slides_worked_example.py) | Matches the lecture slides' own worked example exactly (`Slides/3_system_of_eqs.txt`, slides 33-39) — solution `x=[3,-2.5,7]`. |

## Why this folder exists separately from `templates/` and `algorithms/`

- **`algorithms/`** = clean, canonical, minimally-commented library
  functions — the source of truth.
- **`templates/`** = the same logic wrapped in a full worked example
  (prints, verification, plots) — grab the whole file for a question that
  looks like "solve this general Gauss/LU/power-method problem."
- **`prev-solutions/`** (this folder) = solutions built specifically to
  match the *exact* wording, matrix, and required output of a real
  transcribed past question — useful to see precisely what a graded
  answer needed to include, not just the algorithm in the abstract.

- **`practice/`** = 13 further exam-shaped questions covering everything the
  syllabus can produce that these five did not, with statements kept separate
  in `practice/questions.md` so you can attempt them blind.

If the real exam question turns out to be a close variant of one of these,
start here; otherwise `templates/` is the more general starting point.

## Exact matrices vs. representative ones

All five questions above now use **exact** data: A1's, C1's, and C2's
matrices were reconstructed from what students recorded (`res/A2_Prep.md`
§3–§6) and this repo's outputs match the numbers recorded there; B1's three
systems are also the real ones; B2's `A`, `b1`, `b2` were given directly in
the question, no reconstruction needed. So these files are reconstructions of
what was actually graded, not approximations — worth reading closely for what
a full-credit answer had to *print*, not just compute.

The other two files in `lu_decomposition/` (`lu_q1_doolittle_solve.py`,
`lu_q2_pivoting_existence.py`) predate the C2 transcript and are general
practice, not exam reconstructions. Keep them for drilling; treat
`c2_lu_two_rhs_reuse_factorization.py` as the real one.
