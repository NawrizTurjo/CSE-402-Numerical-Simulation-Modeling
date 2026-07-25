# Online-2 exam context (Lab Assignment 2 / A2)

Live, on-site, timed coding exam. **Date: in ~2 days from 2026-07-22** (check
your actual schedule). Course: CSE401 Numerical Analysis, Simulation and
Modeling, BUET (Nafis Tahmid).

**Official syllabus for this exam:**
> Lab Assignment 2 (A2/B2/C2): Solution of systems of equations: Gauss
> elimination method, Gauss-Jordan elimination method, LU decomposition;
> Eigenvalue decomposition: power method — including all relevant topics
> covered in theory class.

You are section **A2**. Sections **A1, B1, C1, and C2** have already completed their online exams — their transcribed questions and requirements (from A1, B1, C2) are collected in [res/Online Questions.txt](res/Online%20Questions.txt) and fully solved under `code/prev-solutions/`. These serve as the single best signal for what your own exam will look like (same instructor, same syllabus block, staggered by section).

## What to actually do with this repo

**→ Use [code/](code/) on exam day.** It's a tested, ready-to-copy set of
templates covering every topic below. Start with
[code/README.md](code/README.md) and [code/cheatsheet.md](code/cheatsheet.md).

**→ Use [study-guide.md](study-guide.md) *before* exam day.** It's the
study plan for the ~2 days you have left: a 4-step read/retype/diff loop
per algorithm, priority-ordered topics, a timed-mock-exam procedure, and
an exam-day time budget. Read this first, then follow it into `code/`.
Everything in `code/` was built from this folder's slides + past questions +
your friends' collected solutions, cross-checked against the theory slides,
and numerically verified to run correctly (`py -3 <file>.py`, exit 0, outputs
cross-checked against `np.linalg` and against hand-worked slide examples).

`code/` is organized by role, not just by topic:
- `code/algorithms/` — canonical library functions (source of truth).
- `code/docs/` — one `.md` per algorithm, algorithm-level math vs.
  code-level implementation notes, paired 1:1 with `algorithms/`.
- `code/templates/` — full worked exam-answer scripts (grab the whole
  file, swap in the question's matrix).
- `code/complex_cases/` — harder variants (rank deficiency >1, singular
  matrices, ill-conditioning, complex/tied eigenvalues, larger n) with the
  symptom you'd see live and the fix.
- `code/prev-solutions/` — solutions mapped to the actual 4 transcribed
  past questions below, plus a closest available Gauss-Jordan reference
  (the only syllabus topic with no exact transcribed question yet).

The rest of this document is *why* `code/` looks the way it does — the
source material and reasoning, kept here so you can go back to primary
sources if you need more than the templates cover.

## Exam format (from A1, B1, C1, C2's actual online exams)

- **Time limit: 30 minutes, sometimes extended to 35.** Fast. Have your
  templates ready to paste, don't write boilerplate from scratch.
- Typically: an n×n matrix (and RHS vector, if a linear-system question) is
  *given* in the question — you're not deriving it, just plugging it in and
  running your algorithm.
- Consistently required across past questions, regardless of topic:
  1. **Hand-written loop versions for algorithms**: Pivot selection (`argmax`),
     row swaps, forward elimination, and back substitution must use explicit `for`
     loops. Fancy indexing or high-level library functions for these steps are
     disallowed by examiners. All `code/` templates and `code/algorithms/` provide
     hand-written loop implementations with a toggle flag (e.g. `HAND_WRITTEN = True`
     in `code/templates/01_gauss_elimination.py`).
  2. **Print everything**: every pivot chosen, every row swap, the
     augmented/L/U matrix after each step — not just the final answer. Full
     step-by-step annotations are included in template scripts.
  3. **Verify against NumPy** (`np.linalg.solve`/`inv`/`det`/`eig`) and print
     a residual/error norm (`||Ax-b||_2`, `||A-LU||_F`, `||A-LU||_2`, etc.) as the last
     step of almost every question.
  4. A **written/theory question** tacked onto the end of the coding task
     (e.g. "does a zero pivot always mean no solution?", "why is LU cheaper
     for many right-hand sides?"). These come straight out of the theory
     slides — see `code/cheatsheet.md` for a compiled quick-answer bank.
- Eigenvector answers must be **normalized** before comparing to NumPy, and
  a **sign flip** vs. NumPy's output is expected/correct, not a bug (`v` and
  `-v` are equally valid eigenvectors).

### Actual past questions (transcribed, [res/Online Questions.txt](res/Online%20Questions.txt))

1. **Power Iteration** — 4×4 matrix given, implement dominant eigenvalue +
   eigenvector, verify with NumPy. Must normalize before comparing; sign
   flip vs NumPy is fine.
2. **B1's system-of-equations question** (30→35 min) — three 3-variable
   systems given. Gauss elimination with hand-written partial pivoting;
   classify each as unique/no-solution/infinite-solution; print every pivot
   + swap + the post-elimination augmented matrix; for unique solutions
   verify with `np.linalg.solve` + `||Ax-b||_2`; for no-solution, print the
   literal `0x+0y+0z=c` row that proves it; for infinite solutions, pick a
   free variable and back-solve. Written follow-up: does a zero diagonal
   entry always mean no solution? (Answer worked out in
   [res/Src-2/q2_written_answer.md](res/Src-2/q2_written_answer.md) and
   mirrored in `code/algorithms/gauss_elimination.py`'s trailing comment.
   Exact solved match: `code/prev-solutions/gauss_elimination/b1_three_systems_classify_pivot.py`.)
3. **A1's inverse power method question** — exact matrix
   `[[4,1,0,0],[1,3,1,0],[0,1,2,1],[0,0,1,1]]`, find smallest
   eigenvalue/eigenvector via inverse power method using `np.linalg.inv()`
   explicitly (this particular question *named* the library inverse — read
   the actual question wording carefully on the day;
   `code/algorithms/inverse_power_method.py` gives you both the naive
   `np.linalg.inv` version and the "proper" LU-triangular-solve version so
   you're covered either way), verify with NumPy eig. Exact solved match:
   `code/prev-solutions/inverse_power_method/a1_smallest_eigenpair_explicit_inverse.py`.
4. **C2's LU decomposition question** — 3×3 matrix `A` with *two* separate
   RHS vectors `b1`, `b2`. Find the LU decomposition; verify `||A-LU||`;
   solve both systems by forward/backward substitution **reusing the
   already-computed L, U** (do not re-factorize for the second RHS);
   compute an L2 norm; verify both solutions with `np.linalg.solve`; compute
   the residual `||Ax-b||` for both. Written follow-up: which part of the
   computation was reused, and why is recomputing L, U for the second RHS
   unnecessary? (Answer: L, U depend only on A, not b — factor once,
   O(n³); each RHS after that is just two O(n²) triangular solves. Full
   argument in the trailing comment of the solved file.) Exact solved match:
   `code/prev-solutions/lu_decomposition/c2_lu_two_rhs_reuse_factorization.py`.

### Question banks used to build the templates

Two friends (`res/Src-2`, `res/Src-14`) independently wrote out full
coding-question banks matching this exact syllabus, based on the same
lecture:
- [res/Src-2/coding_questions.md](res/Src-2/coding_questions.md) — systems
  of equations bank (robust solver, LU "two-step dance", inversion via LU,
  "determinants for free", Gauss-Jordan).
- [res/Src-2/eigen_coding_questions.md](res/Src-2/eigen_coding_questions.md)
  — eigenvalue bank (power method, inverse power method, Hotelling
  deflation, eigenvalue-decomposition verification).
- [res/Src-14/lu_practice_questions.md](res/Src-14/lu_practice_questions.md)
  — 5 exam-length (~30-35 min) LU practice questions with a full worked
  answer key, including flop-count written questions.

These are worth skimming directly if you have spare prep time — they're
more detailed than the compressed cheatsheet in `code/`.

## Syllabus content (from the two slide decks in [Slides/](Slides/))

Extracted to text via `pdftotext -layout` at
[Slides/3_system_of_eqs.txt](Slides/3_system_of_eqs.txt) (71 slides) and
[Slides/4_eigen_decomposition.txt](Slides/4_eigen_decomposition.txt)
(63 slides) if you want to grep the original wording/examples.

**Deck 3 — Systems of equations:**
- Gauss elimination: forward elimination + back substitution, worked
  rocket-velocity curve-fit example.
- Pitfalls: division by zero (immediate, and the "sneaky" version where a
  pivot only becomes zero after a step of elimination), round-off error
  (worked example showing 6-digit vs 5-digit precision producing 50% vs 5%
  error from the same tiny pivot).
- Partial pivoting as the fix for both pitfalls; worked example showing the
  same system going from 50% error (naive) to exact (pivoted).
- Determinant "for free" from elimination: row ops preserve det, triangular
  → product of diagonal, each swap flips sign.
- Gauss-Jordan: eliminate above *and* below the pivot, normalize every
  pivot row to 1 → augmented matrix reaches `[I|x]` directly.
- LU decomposition: `U` = the Gauss elimination result, `L` = the
  multipliers you'd otherwise throw away, stored at their `(i,k)` slot.
  Full clock-cycle cost derivation showing **LU costs the same as Gauss for
  one solve**, but wins by ~n× when solving for many right-hand sides with
  the same `A` (this is the point of the whole method) — includes a
  table showing Gauss-for-inverse costing ~2500× more than LU-for-inverse
  at n=10000.
- Matrix inverse via LU: solve `Ax_j = e_j` for every column of `I`, reusing
  one factorization.

**Deck 4 — Eigenvalue decomposition / power method:**
- Eigenvectors/eigenvalues built up from "a matrix is a machine that eats a
  vector" intuition, worked 2×2 example (`[[2,1],[1,2]]`, eigenvalues 3, 1).
- Derivation of `A = VΛV⁻¹` from `Avᵢ=λᵢvᵢ` (not just stated) and why it
  makes `Aᵏ` cheap (`Aᵏ = VΛᵏV⁻¹`, the middle `V⁻¹V` pairs cancel
  telescopically).
- Power method: full derivation of why the dominant eigenvalue wins
  (`Aᵏx₀ = Σ cᵢλᵢᵏvᵢ`, factor out `λ₁ᵏ`, other terms → 0), step-by-step
  worked 2×2 and 3×3 examples with iteration tables.
- Finding the smallest eigenvalue: inverse trick (`λ` eigenvalue of `A` ⟺
  `1/λ` eigenvalue of `A⁻¹`), full worked 4×4 example with iteration table.
- Finding intermediate eigenvalues: Hotelling's deflation — "mute" the
  dominant eigenvector's contribution via `A₂ = A - λ₁v̂₁v̂₁ᵀ`, works because
  eigenvectors of a symmetric matrix are orthogonal. Full derivation of why
  this leaves the other eigenvectors untouched.

## Folder inventory

```
Online-2/
├── agent.md                     <- this file
├── Slides/                      original lecture PDFs + extracted .txt
├── code/                        <- USE THIS ON EXAM DAY (tested templates)
│   ├── algorithms/               canonical library functions, one per topic
│   ├── docs/                     algorithm-level vs code-level .md per topic
│   ├── templates/                full worked exam-answer scripts
│   ├── complex_cases/            harder variants + symptom/fix notes
│   ├── prev-solutions/           solutions mapped to the 4 real transcribed past questions (A1, B1, C2, Power Iteration)
│   ├── README.md, cheatsheet.md
└── res/
    ├── Online Questions.txt     transcribed real online-exam questions (A1, B1, C2, Power Iteration)
    ├── Src-2/                   friend's collected code + 2 question banks
    │   ├── coding_questions.md, eigen_coding_questions.md
    │   ├── q2_written_answer.md (zero-pivot theory answer, worked out)
    │   ├── Gauss_Elimination/, Gauss_Jordan/, LU_Decomposition/,
    │   │   Eigenvalue_Decomposition/  (per-topic solved question files)
    ├── Src-14/                   friend's collected code (highest quality —
    │   ├── lu_decomposition_theory.md      most of code/ is built on this)
    │   ├── lu_practice_questions.md        (5 exam-style Qs + answer key)
    │   ├── numpy_cheatsheet.md              (merged into code/cheatsheet.md)
    │   ├── LU/, gauss-jordan/, power/
    ├── Src-26/                   another friend's collected code (rougher,
    │                             some incomplete/buggy functions — used only
    │                             for cross-checking, not as a source)
    └── Src-29/                   another friend's collected code (rougher,
                                  two files empty; not used as a source)
```

`Src-14`'s code was the cleanest and most correct of the four collections —
consistent style, explicit hand-written pivoting, thorough NumPy
verification, and an accompanying theory cheat sheet that matches the
slides almost line-for-line. It's the primary basis for `code/`.
`Src-26`/`Src-29` contain real bugs (e.g. `np.max` used where
magnitude-based `np.argmax(np.abs(...))` was needed, incomplete function
bodies, unused/broken duplicate definitions) — don't copy from them
directly without testing.

## One correctness bug found and fixed while building `code/`

`code/templates/06_deflation_intermediate_eigen.py`'s original demo used
`x0=[1,1,1,1]` for a block-symmetric matrix. After deflating away the
dominant eigenvalue (10), the surviving eigenvector for the next eigenvalue
(6) is `[1,-1,0,0]`-direction — and `[1,1,1,1]` has **zero component** along
it. Power iteration silently converged to the *third* eigenvalue (4) instead
of the second (6), with no error or warning. Fixed by using a generic,
non-symmetric `x0=[1,2,3,-1]`, verified against `np.linalg.eig` to recover
all four eigenvalues (10, 6, 4, 2) correctly. This exact trap is documented
in `code/algorithms/deflation.py`, `code/docs/deflation.md`, and
`code/complex_cases/README.md` — worth remembering live in the exam if your
power method "converges instantly" to a suspiciously-fast wrong answer.
