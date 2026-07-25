# Previous solutions — mapped to real past questions

`res/Online Questions.txt` transcribes **4** real online-exam
questions from past sections (B1/A1/C2).
This folder maps each one to a working, tested solution, plus the closest
available reference for Gauss-Jordan (in the syllabus but not among the 4 transcribed questions).

| Real question (from `res/Online Questions.txt`) | Solved here | Notes |
|---|---|---|
| **Power Iteration**, general 4×4, verify vs NumPy | [power_method/q_dominant_eigenpair_4x4.py](power_method/q_dominant_eigenpair_4x4.py) | The transcript doesn't record the exact matrix used — this is `res/Src-14`'s worked version on a representative 4×4 (block-diagonal, eigenvalues 10/6/4/2). Swap in whatever matrix the real question gives. |
| **B1's system-of-equations question** — 3 systems, hand-written partial pivot, classify unique/no-solution/infinite, print every pivot+swap+post-elimination matrix, verify unique case vs NumPy | [gauss_elimination/b1_three_systems_classify_pivot.py](gauss_elimination/b1_three_systems_classify_pivot.py) | Exact match to the transcribed question's requirements, including the written-question answer (zero-diagonal-entry theory) in the trailing comment. |
| **A1's inverse power method** — exact matrix `[[4,1,0,0],[1,3,1,0],[0,1,2,1],[0,0,1,1]]`, explicitly using `np.linalg.inv()`, verify vs NumPy | [inverse_power_method/a1_smallest_eigenpair_explicit_inverse.py](inverse_power_method/a1_smallest_eigenpair_explicit_inverse.py) | Matches exact matrix in prompt using `np.linalg.inv()`. Smallest eigenvalue ≈ `0.254719`. |
| **C2's LU decomposition question** — 3×3 matrix `A`, two RHS `b1, b2`, reuse L,U factorization, verify norms `||A-LU||` and `||Ax-b||_2`, verify vs `np.linalg.solve()` | [lu_decomposition/c2_lu_two_rhs_reuse_factorization.py](lu_decomposition/c2_lu_two_rhs_reuse_factorization.py) | Solves both systems by reusing computed L, U, calculates matrix residual norms, verifies with NumPy, and details why recomputing L, U is unnecessary. |
| *(not literally asked, but in the A2 syllabus)* Gauss-Jordan | [gauss_jordan/slides_worked_example.py](gauss_jordan/slides_worked_example.py) | Matches the lecture slides' own worked example exactly (`Slides/3_system_of_eqs.txt`, slides 33-39) — solution `x=[3,-2.5,7]`. |

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

If the real exam question turns out to be a close variant of one of these,
start here; otherwise `templates/` is the more general starting point.
