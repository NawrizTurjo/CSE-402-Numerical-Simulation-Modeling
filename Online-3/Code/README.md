# CSE 402 Online-3 — Reference Codebase

Reusable, exam-ready Python for: RNG (LCG, Middle-Square), statistical
tests (Chi-Square, K-S, autocorrelation/runs), Monte Carlo methods
(integration, hit-or-miss, Metropolis-Hastings), inverse-transform
variate generation, and discrete-event single-server-queue simulation.

For the "what to study, in what order" guide, read
[`../STUDY_GUIDE.md`](../STUDY_GUIDE.md). This file is just setup +
folder map.

## Setup

```bash
cd Code
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Only `numpy`, `scipy`, `matplotlib` are used, and only `scipy.stats` is
load-bearing (for exact critical values / p-values in the testing/
modules — everything else is pure Python standard library: `math`,
`random`, `heapq`, `itertools`).

## Running things

Every module is both importable AND directly runnable. Run as a module
from inside `Code/` so the package-relative imports (`from rng.lcg import
...`) resolve:

```bash
python -m rng.lcg
python -m rng.middle_square
python -m testing.chi_square_test
python -m testing.ks_test
python -m testing.independence_test
python -m monte_carlo.core
python -m monte_carlo.integration
python -m monte_carlo.metropolis_hastings
python -m variate_generation.inverse_transform
python -m simulation.single_server_queue
python -m solutions.a1_buffons_needle
python -m solutions.c1_middle_square_investigation
python -m solutions.practice_src26.p1_sphere_volume
# ...and so on for p2-p6
```

If the exam only lets you submit a single file, copy the specific
function(s) you need straight out of the relevant module — everything
here is self-contained (no module depends on anything outside this
`Code/` folder, `math`/`random`/`heapq`/`itertools`, and `scipy.stats`
where noted).

## Folder map

```
rng/                    Pseudo-random number generators
  common.py                shared cycle-detection + binning helpers
  lcg.py                   Linear Congruential Generator + period theory
  middle_square.py         Middle-Square method (C1 question), any digit
                            width via `digits=`, plus a Weyl-sequence
                            repair (`middle_square_weyl`) for its short-cycle
                            weakness

variate_generation/
  inverse_transform.py     Uniform(0,1) -> any distribution via F^-1

testing/                 Statistical tests FOR a sequence of numbers
  chi_square_test.py        uniformity (1-D bin counts)
  ks_test.py                 uniformity (empirical CDF vs diagonal)
  independence_test.py       autocorrelation test + TWO runs tests
                              (above/below-mean, and up/down) -- they
                              catch different kinds of dependence, see
                              the module docstring

monte_carlo/
  core.py                  THE generic estimator -- read this first
  integration.py            1-D integration + hit-or-miss pi
  metropolis_hastings.py    MCMC sampling from an unnormalized density

simulation/
  single_server_queue.py   ONE DES engine covering every queueing variant
                            (N-delays / T_max / drain-to-completion,
                            1 or c servers, optional balking, optional trace)

solutions/                Worked answers, built on the modules above
  a1_buffons_needle.py       A1 (online exam)
  c1_middle_square_investigation.py   C1 (online exam), all 4 tasks
  practice_src26/            practice-problems.md #1-#6
  practice_generated/        ../../Practice/PRACTICE_QUESTIONS.md #R1,R2,S1,V1,M1,M2,MH1
```

## Design principle (read this before the exam)

Every folder above picks ONE canonical, parameterized implementation of
its sub-problem instead of near-duplicate variants. Concretely:

- **`monte_carlo/core.py`** has exactly one estimator function. Every
  Monte Carlo problem you'll see (probability, expected value, integral,
  volume) is answered by writing a different one-line `trial_fn` and
  calling that same function — see its module docstring for why.
- **`simulation/single_server_queue.py`** has exactly one simulation
  loop. Stop-after-N-delays, stop-at-T_max, run-to-completion, multiple
  servers, and balking are all keyword arguments on `simulate_ssq`, not
  separate scripts.
- **`rng/common.py`**'s `find_cycle` is shared by both `rng/lcg.py` and
  `rng/middle_square.py` because cycle detection is the same algorithm
  regardless of which recurrence produces the sequence.

So: understand `monte_carlo/core.py` and `simulation/single_server_queue.py`
deeply (they're short), and every "new" problem in the exam is
recognizing which shape it is and writing a few lines around the shared
engine — not writing a new loop from scratch under time pressure.
