# Study guide — how to actually use `code/` before the exam

Goal is not "read all the files." Goal is: **when a matrix appears on
screen with a 30-minute clock running, your hands know what to type before
your brain finishes reading the question.** That only comes from active
practice, not reading. This guide is the practice plan.

You have ~2 days. Budget accordingly — this plan assumes 2 focused
sessions of ~2-3 hours each, not all-day cramming.

---

## The core skeleton — memorize this cold, first

Almost everything in `code/algorithms/` is five primitives recombined.
Get these into your fingers before anything else; every algorithm below is
just these pieces in a different order.

```python
# 1. hand-written pivot search (largest |entry| in column k, rows k..n-1)
p = k
for i in range(k + 1, n):
    if abs(A[i, k]) > abs(A[p, k]):
        p = i

# 2. hand-written row swap
for j in range(n):
    A[k, j], A[p, j] = A[p, j], A[k, j]

# 3. elimination step (zero out column k below the pivot)
for i in range(k + 1, n):
    m = A[i, k] / A[k, k]
    A[i, k:] -= m * A[k, k:]

# 4. forward substitution (Lz = b, L unit-lower-triangular)
z = np.zeros(n)
for i in range(n):
    z[i] = (b[i] - L[i, :i] @ z[:i]) / L[i, i]

# 5. backward substitution (Ux = z)
x = np.zeros(n)
for i in range(n - 1, -1, -1):
    x[i] = (z[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

# 6. eigenvalue estimate, magnitude-based, sign preserved
lam_est = y[np.argmax(np.abs(y))]
```

**Test yourself right now, before reading further:** close this file, open
a blank `scratch.py`, and type all six from memory. Time it. If it takes
more than ~4 minutes or you get any of them wrong (especially #6 —
`np.max(y)` instead of `y[np.argmax(np.abs(y))]` is the single most common
mistake in the raw collected code, see `code/docs/power_method.md`), that's
your first study session's actual job, not "reading `algorithms/`."

---

## The 4-step loop for each algorithm

Do this once per topic, topics in priority order (see below). ~25-35 min
per topic the first time, faster on repeat passes.

1. **Read the algorithm-level section** of `code/docs/<topic>.md` — the
   math, not the code. Can you re-derive *why* the method works, in your
   own words, without looking? (e.g. "power method converges because
   `A^k x0` is dominated by the largest `lam_i^k` term as `k` grows.") If
   not, re-read until you can say it out loud without the doc open.
2. **Read `code/algorithms/<topic>.py` once**, slowly. Note anything that
   surprised you — that's usually the "code level" section of the doc
   explaining exactly that surprise.
3. **Close the file. Open a blank file. Retype the whole thing from
   memory**, function by function. Don't peek. When you get stuck, that's
   the exact spot to re-read — not the whole file, just that line.
4. **Diff your version against the original** (`code/algorithms/<topic>.py`).
   Any difference that changes behavior (not just variable names) is a gap
   — note it, redo step 3 for that function only, tomorrow.

This is slower than reading through everything once, but reading-only
retention for "can I produce this under time pressure" is close to zero —
you will recognize the code as familiar and still fail to reproduce it
blank-page, which is exactly the exam's format.

---

## Priority order (study in this order, stop early if time runs out)

Ranked by (a) syllabus weight, (b) how many of the 5 real transcribed
questions touched it, (c) how much of every other topic it's built from.

1. **Gauss elimination + partial pivoting** — `docs/gauss_elimination.md`.
   Foundation for everything else (LU literally reuses this loop). One of
   the 5 real past questions was exactly this.
2. **Power method** — `docs/power_method.md`. One of the 5 real past
   questions. Short, but the `argmax(abs(.))` trap is the single most
   common real bug — drill it until it's automatic.
3. **LU decomposition** (naive + `PA=LU` + solve/det/inverse) —
   `docs/lu_decomposition.md`. Biggest syllabus surface area (4+ sub-skills
   in one topic), explicitly named in the syllabus. One of the 5 real past
   questions (C2 — two RHS, reuse L/U) — high risk of appearing again.
4. **Inverse power method** — `docs/inverse_power_method.md`. One of the 5
   real past questions, and it's short once LU (#3) is solid — it's just
   "power method, but solve instead of multiply."
5. **Gauss-Jordan, including the matrix inverse via `[A|I]`** —
   `docs/gauss_jordan.md`. Small delta on top of #1 (normalize the pivot row +
   eliminate in both directions), and the inverse is the *same loop* with the
   right-hand block widened from 1 column to n. Cheap to add once #1 is solid,
   and it's now confirmed real too (B2 — augmented `[A|b1|b2]`, then the same
   sweep re-run on `[A|I]` for the inverse). Drill with
   `prev-solutions/gauss_jordan/b2_two_rhs_and_inverse.py`,
   `practice/extra/e2`, and `practice/a2prep/p1a`, `p1b`.
6. **Deflation** — `docs/deflation.md`. Builds directly on #2. Watch the
   `x0` blind-spot trap (documented in the doc) — it's the kind of bug
   that looks like success (a number pops out) but is silently wrong.
7. **Eigenvalue decomposition (`A=VΛV⁻¹`)** — `docs/eigen_decomposition.md`.
   Shortest of the seven, mostly `np.linalg.eig` plumbing — do this last,
   it won't take long once you've internalized eigenvector-column
   conventions from #2.

If you only have time for a partial pass before the exam, stop after #4 —
that covers both real past-question topics plus the biggest syllabus item.
\#5-7 are each under 15 minutes to drill once #1-4 are solid.

---

## Timed mock exams (do at least 3, spread across your remaining time)

Reading/retyping builds the *pieces*. Mock exams build the *assembly under
pressure*, which is the actual skill being tested. Don't skip this even if
short on time — it matters more than a 4th read-through of any single doc.

**Procedure**, repeat with a different question each time:

1. Pick a source, in this priority order for realism:
   - `code/practice/questions.md` — **13 ready-made questions, statements
     only, no solutions visible.** This is the intended source: each one is
     exam-shaped, timed at 25–35 min, and has a full worked solution in
     `code/practice/a2prep/` or `code/practice/extra/` to grade yourself
     against afterwards.
   - `code/prev-solutions/` — the five questions that were actually set. Read
     the problem statement at the top of the file, then close it.
   - `code/complex_cases/` — harder variants; use once the basics feel solid.
2. **Set a 30-minute timer.** Close every file except a blank editor.
3. Solve it cold: write the function(s), run against your own hand-picked
   matrix, print every intermediate step the past questions have asked for
   (pivots, swaps, L/U per column, augmented matrix per step), verify
   against `np.linalg.solve`/`inv`/`det`/`eig`, and answer the kind of
   written follow-up question listed in `code/cheatsheet.md`.
4. **Stop at 30 minutes**, whether done or not — this is the actual
   constraint you'll face.
5. Compare against the reference solution. Grade yourself: did the printed
   output match what past questions explicitly asked for (not just "did I
   get the right number")? Note the ONE thing that cost you the most time,
   and drill just that thing before your next mock.

Suggested mock rotation if you do exactly 3 (all from
`code/practice/questions.md`):
- Mock 1: **E7** — B1's real question shape, one difficulty step up
  (4×4, two free variables). Classification under time pressure is where
  careless marks go.
- Mock 2: **E3** — C2's real question shape, but on a matrix where naive LU
  actually fails. If the exam matrix needs pivoting and you have only drilled
  the no-pivot version, this is the run that saves you.
- Mock 3: **E4** — covers both eigenvalue questions that have really been set
  (C1's power method and A1's inverse power method) in one sitting.

If you get a fourth: **P5 or E5** (deflation — the highest-risk untested topic,
and the last one taught), or pick one from `complex_cases/` at random as a
stress test. Also worth a cold run once, since it's real: B2 itself
(`prev-solutions/gauss_jordan/b2_two_rhs_and_inverse.py`) — augmented
`[A|b1|b2]` solved by Gauss-Jordan, then the same sweep repurposed on `[A|I]`
for `A⁻¹`.

---

## Exam-day time budget (30 min; stretch to 35 if extended)

Past questions are dense — budget time explicitly, don't let one section
eat the whole clock.

| Time | Task |
|---|---|
| 0:00–0:03 | Read the question fully once. Identify: which algorithm, what output format is explicitly required (prints? verification? written question?), what matrix/vector is given. |
| 0:03–0:05 | Open the matching file (check `prev-solutions/README.md` first, then `templates/`). Paste in / adapt the given matrix. |
| 0:05–0:20 | Core algorithm. Run it early and often — don't write all 40 lines blind and run once at the end. Run after forward elimination, run after adding back-substitution, etc. |
| 0:20–0:25 | Required prints (pivots/swaps/intermediate matrices) + NumPy verification + residual/error norm. |
| 0:25–0:30 | Written/theory question at the end (see `code/cheatsheet.md` — most of these are one paragraph, don't overthink, they're graded on hitting the right concept, not prose quality). |
| buffer | If ahead of schedule: double check sign conventions, re-read the question for a requirement you might've skipped (a very common past-question trap: "print every row swap" gets forgotten under pressure). |

If something isn't converging or produces `nan`/garbage and you don't
immediately see why, don't debug blind — check `code/complex_cases/README.md`'s
symptom table first (30 seconds), it's a very short list and covers the
likely causes (singular matrix, tied eigenvalues, complex eigenvalues,
oscillating power method).

---

## Pre-submit checklist (30 seconds, every question)

- [ ] Used hand-written pivot/swap loops, not `np.argmax`/fancy indexing,
      unless the question said that was fine.
- [ ] **Every norm, residual and `A@x` that is part of the ANSWER is
      hand-coded**, with the `np.linalg` value printed beside it as a check —
      not instead of it. (`np.linalg.norm(x)` to normalize your own
      eigenvector, `A @ x` inside a residual, and `np.linalg.norm(A - L@U)`
      all count as computing your own answer with a library. See
      `code/algorithms/manual_ops.py`.)
- [ ] Eigenvalue estimates use `y[np.argmax(np.abs(y))]`, never `np.max(y)`.
- [ ] Eigenvectors normalized before comparing to NumPy; sign-flip check
      (`if np.dot(v, v_np) < 0: v_np = -v_np`) present if comparing.
- [ ] If `PA=LU`: solved `Lz = P@b`, not `Lz = b`.
- [ ] Printed everything the question explicitly asked for (re-read the
      question once more specifically hunting for "print X" instructions —
      easy to lose under time pressure).
- [ ] NumPy verification line + residual/error norm present.
- [ ] Written question answered, even briefly — a short correct answer
      beats a blank one.

---

## What NOT to do

- Don't try to memorize entire `templates/*.py` files verbatim — memorize
  the six primitives above and the *shape* of how each topic assembles
  them; reconstruct the rest live. Verbatim memorization is brittle (one
  forgotten line and you're stuck); understanding the assembly means you
  can always rebuild a dropped piece.
- Don't spend disproportionate time on `code/complex_cases/` before the
  basics (priority list above) are solid — those are for recognizing a
  symptom quickly if you're unlucky, not core exam content.
- Don't skip the timed mocks in favor of more reading. Reading gives you
  recognition; only timed practice gives you production speed, which is
  the actual bottleneck in a 30-minute live-coding exam.
- Don't cram new topics the night before — a solid pass on `algorithms/`
  1-4 plus one full timed mock beats a shallow pass on all 7 topics with
  no mocks.
