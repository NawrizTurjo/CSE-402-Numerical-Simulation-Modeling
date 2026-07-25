# Complex / edge cases

Harder variants of the core algorithms — things past exam questions
haven't (yet) asked directly, but that are one unlucky matrix away from
showing up, or that are worth recognizing on sight if your code produces a
suspicious result live in the exam. Every script here runs standalone
(`py -3 <file>.py`) and prints its own explanation.

| File | Scenario | Symptom if you hit it unprepared |
|---|---|---|
| [multiple_free_variables.py](multiple_free_variables.py) | Rank deficiency > 1 (more than one free variable) | Past exams only ever needed 1 free variable — this generalizes `classify_and_solve` to any number, using the "zero row `i` → free variable `x_i`" insight |
| [singular_and_rank_deficient.py](singular_and_rank_deficient.py) | Zero leading principal minor (naive LU) vs. genuinely singular matrix (no pivoting can fix it) | Naive LU silently fills with `inf`/`nan`, no exception raised |
| [ill_conditioned_systems.py](ill_conditioned_systems.py) | Small residual, large actual error (Hilbert matrix) | `||Ax-b||` looks fine, `x` is still wrong — must check `kappa(A)` |
| [complex_conjugate_eigenvalues.py](complex_conjugate_eigenvalues.py) | Dominant eigenvalue pair is complex (non-symmetric `A`) | Power method estimates cycle in a repeating pattern, never converge |
| [equal_magnitude_eigenvalue_tie.py](equal_magnitude_eigenvalue_tie.py) | Two real eigenvalues tied in magnitude, opposite sign | Power method oscillates between two values, neither of which is the answer; fix via power method on `A@A` |
| [larger_systems_generic_n.py](larger_systems_generic_n.py) | n=6, n=8 random systems (not toy 3x3/4x4) | Confirms every `algorithms/` function generalizes past hardcoded-feeling small examples |

## How to use this folder on exam day

You won't have time to read these live. Skim them **once** before the
exam so the *symptoms* are familiar — if your power method result looks
like it's cycling instead of converging, or your LU factorization fills
with `nan`, you'll recognize which of these you've hit and know the fix
immediately instead of debugging from scratch under a 30-minute clock.

The one genuinely reusable code upgrade here (not just a diagnostic) is
`multiple_free_variables.py`'s generalized `classify_and_solve` — if a
question gives you a system with more than one degree of freedom, that's
the version to reach for instead of `templates/01_gauss_elimination.py`'s
single-free-variable version.
