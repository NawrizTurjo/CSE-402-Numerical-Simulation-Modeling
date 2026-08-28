# AGENTS.md — Online-3 (CSE 402 Numerical Methods)

Read this file first. It tells you where everything is so you don't
need to re-read the slide PDFs or re-derive the reference code — both
are already done.

## Context

30-minute **live, onsite, offline** coding exam. Syllabus
([Res/syllabus.txt](Res/syllabus.txt)): discrete-event simulation models,
steps in a simulation study, model validation/verification, random
number generation, Monte Carlo methods including Metropolis-Hastings.
Two subsections (A1, C1) have already sat the exam; their remembered
questions are in [Res/questions.txt](Res/questions.txt) (A1 = Buffon's
Needle, C1 = Middle-Square RNG investigation — not official, reconstructed
from memory). The exam allows bringing template code from home, so
[Code/](Code/) is a pre-built, tested, reusable library to bring in.

**No internet access during the exam** — do not attempt any WebFetch/
WebSearch, everything needed is already local.

## Where things are

| Need | Location |
|---|---|
| Study plan / topic-by-topic guide | [STUDY_GUIDE.md](STUDY_GUIDE.md) |
| Slide content, DES/SSQ deck (154 slides) as text | [Slides/5_simu_des_ssqs.md](Slides/5_simu_des_ssqs.md) |
| Slide content, RNG/Monte Carlo deck (82 slides) as text | [Slides/6_rng_mcs.md](Slides/6_rng_mcs.md) |
| Reusable, tested code library | [Code/](Code/) — see [Code/README.md](Code/README.md) for the folder map |
| Worked exam-question answers | [Code/solutions/](Code/solutions/) |
| Practice-problem answers (Src-26 set) | [Code/solutions/practice_src26/](Code/solutions/practice_src26/) |
| More practice questions (one per syllabus pillar, Q+A) | [Practice/PRACTICE_QUESTIONS.md](Practice/PRACTICE_QUESTIONS.md), code in [Code/solutions/practice_generated/](Code/solutions/practice_generated/) |
| Original raw slide PDFs (fallback only — the .md above already captures everything from them) | [Slides/*.pdf](Slides/) |
| Collected friends' code (raw material the Code/ library was built from) | [Res/Src-14](Res/Src-14/), [Res/Src-17](Res/Src-17/), [Res/Src-26](Res/Src-26/), [Res/Src-29](Res/Src-29/) |

Do not re-open the slide PDFs unless the `.md` transcripts are missing
something specific — they were produced by a careful page-by-page read
and already contain every formula, algorithm, and worked example.

## Important note on scope

The RNG/Monte Carlo slide deck ([Slides/6_rng_mcs.md](Slides/6_rng_mcs.md))
does **not** contain the Middle-Square method or the Metropolis-Hastings
algorithm, even though the syllabus explicitly names Metropolis-Hastings
and C1's question is about Middle-Square. Both were evidently covered
live in class beyond what's in these two PDFs. Both are still fully
implemented, tested, and documented in [Code/rng/middle_square.py](Code/rng/middle_square.py)
and [Code/monte_carlo/metropolis_hastings.py](Code/monte_carlo/metropolis_hastings.py)
(reconstructed from the collected friends' code, which is consistent
across all four sources).

## If asked to help during exam prep or the exam itself

1. Identify which syllabus topic the question is testing.
2. Point to (or directly reuse) the matching function in `Code/` —
   don't write a new implementation from scratch; the library already
   covers: LCG (with period/max-period checking), Middle-Square (any
   digit width via `digits=`, plus a Weyl-sequence repair for its
   short-cycle weakness), Chi-Square test, K-S test, autocorrelation
   test, TWO runs tests (above/below-mean and up/down — they catch
   different dependence, see `Code/testing/independence_test.py`),
   inverse-transform variate generation, generic Monte Carlo estimation
   (probability/expected-value/integral/volume — all one function, see
   `Code/monte_carlo/core.py`), Metropolis-Hastings, and a single unified
   single/multi-server queue DES engine
   (`Code/simulation/single_server_queue.py`) that covers stop-by-N-delays,
   stop-by-T-max, run-to-completion, and balking as keyword arguments.
   Two support-only packages are also available but never required:
   `Code/viz/` (matplotlib plots for RNG diagnostics and Q(t)/B(t) — use
   only if a question explicitly asks for a plot) and `Code/cheatsheets/`
   (pure Python/numpy/random/scipy syntax reference, no simulation logic).
3. If the exam's actual question doesn't match any existing shape,
   adapt the closest module rather than starting over — see
   `Code/README.md`'s "Design principle" section for how the modules are
   meant to be extended.
