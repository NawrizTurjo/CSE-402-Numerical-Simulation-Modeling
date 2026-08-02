# AGENT.md — CSE-402 Online Lab Exam Prep Handoff

**Working directory: `E:\4-1 Course Works\CSE-402-Numerical\Online-1`**
Treat this as the project root for all paths below. Do NOT re-explore the codebase from scratch — this file already contains the findings of a full audit. Read a file only if you need to change it.

## Context
CSE-402 Numerical Methods online lab exam (root-finding: Bisection, False Position, Newton-Raphson, Bairstow). Exam-day setup: student is allowed to bring pre-written cheatsheet/code files. Goal of prior session: (1) verify `prev-section/` practice solutions actually match this term's leaked/practiced question set, (2) audit the whole `codes/` folder for consistency/correctness since it'll be leaned on live during the exam.

## Directory map
```
Online-1/
  res/questions.md              <- formatted exam-style question bank (A1, B1, B2, C1, C2) used to validate prev-section/
  references/                    <- AI-extracted slide-content, ground truth for formulas/conventions
    numerical-methods-errors-reference.md       <- Lab 1: approx/round-off/truncation errors, ea/eps_a definitions
    numerical-methods-rootfinding-reference.md  <- Lab 1 cont'd: bisection/false-position/Newton-Raphson (Bairstow explicitly out of scope, no slide deck for it yet)
  codes/
    README.md                    <- exam-day navigation + strategy (accurate, not modified)
    cheatsheets/00_exam_cheatsheet.py   <- "START HERE" all-in-one file (FIXED this session, see below)
    basic/
      01_errors_and_approximations.py  <- safe, has UTF-8 stdout wrapper
      02_visualization_and_plotting.py <- safe (unicode only in docstrings/plot labels, never printed)
      07_error_tradeoff_plotter.py     <- safe, has UTF-8 wrapper
      scanner.py                       <- safe, pure ASCII; EXTENDED this session (coarse_scan/safe_eval/classify_and_verify)
    methods/
      03_bisection_and_false_position.py  <- FIXED this session
      04_newton_raphson.py                <- FIXED this session
      05_bairstow_method.py               <- FIXED this session
    prev-section/                  <- modular per-question solutions, matched against res/questions.md
      a1-diode_equation.py         <- Newton-Raphson, diode eqn — matches exam A1 exactly
      b1-multi_root_scanner.py     <- Bisection multi-root scan — matches exam B1 exactly
      b2-pathological_functions.py <- REWRITTEN this session to match exam B2 exactly (see below)
      c1-log_scale_convergence.py  <- Bisection vs False Position, ln(x), log-scale — matches exam C2
      c2-first_match_scanner.py    <- scan + False Position on first interval — matches exam C1 pattern
      question-bank.md             <- theory + worked solutions for all 5, useful to copy explain-paragraphs from
    prev_year_solves/06_prev_batch_all.py  <- NOT touched this session (has its own UTF-8 wrapper, already safe; user explicitly deferred work here)
    docs/STUDY_GUIDE.md                       <- pre-existing, NOT modified
    docs/NEWTON_RAPHSON_DEEP_DIVE.md          <- pre-existing, NOT modified
    docs/BAIRSTOW_DEEP_DIVE.md                <- pre-existing, NOT modified
    docs/SLIDE1_ERRORS_APPROX_STUDY_GUIDE.md  <- NEW this session (see below)
    docs/SLIDE2_ROOTFINDING_STUDY_GUIDE.md    <- NEW this session (see below)
```

Note: file *names* in `prev-section/` don't line up 1:1 with exam labels — `c1-log_scale_convergence.py`'s content actually answers exam question **C2** (ln(x) log-scale bisection-vs-falseposition), and `c2-first_match_scanner.py`'s content answers exam question **C1** (scan + false position on first interval). Content is correct; only the filenames are swapped relative to `res/questions.md` labels. Don't rename, just be aware when picking a template in the exam.

## Work done this session

### 1. Verified prev-section/ against res/questions.md
All 5 questions (A1, B1, B2, C1, C2) have a matching template. Only gap found: **B2** — the practice code was a different (older) variant of the pathological-function question, missing several requirements the current exam version demands.

### 2. Rewrote `prev-section/b2-pathological_functions.py`
Now matches `res/questions.md` B2 exactly:
- Function `f(x) = (x-2.5)^2*(x+1.052) / (x-3.551)`, domain `[-2,5]`, step `0.1`
- Direct-root candidate detection: `|f(x)| <= 1e-8`
- Sign-change interval candidate detection (separate from direct)
- Bisection run per interval candidate; **terminates that run** if an undefined value is hit mid-bisection (guards the asymptote at x=3.551)
- Acceptance filter applied literally as spec states: accept only if `|x_r| <= 1e-6` (this is the exam's literal wording — most candidates will show REJECT, that's expected/correct behavior, not a bug)
- Prints a candidate verdict table: type (direct/interval), value, f-value, ACCEPT/REJECT
- 3-point written explanation (double-root no sign change / asymptote fake sign change / why check residuals)
- Verified by running it: double root at x=2.5 correctly flagged+rejected, real root near x=-1.052 found (~-1.05200), asymptote interval correctly shows blown-up f-value (garbage root demo).

### 3. Full codebase consistency audit (`codes/` folder)
Found two real defects, both now fixed and verified by actually executing the files (not just reading):

**Bug — `cheatsheets/00_exam_cheatsheet.py`, `bisection()` and `false_position()`:**
Had an inverted bracket-update line:
```python
xl, xu = (xl, xr) if f(xl)*fxr > 0 else (xr, xu)   # WRONG — contradicted own docstring
```
Confirmed empirically wrong: on `f(x)=x^3-x-1`, `[1,2]` (true root ≈1.3247179572), old code converged to **1.5** (wrong). Fixed to the correct explicit form (same style already used by `false_position_illinois()` in the same file and by `methods/03`):
```python
if f(xl)*fxr < 0:
    xu = xr
else:
    xl = xr
```
Re-verified after fix: converges to 1.32471796, correct. `bairstow_all_roots()`, `newton_raphson()` in the same file were already correct — only these two functions had the bug.

**Crash risk — Unicode in `print()` calls without a UTF-8 stdout wrapper:**
`cheatsheets/00_exam_cheatsheet.py`, `methods/03_bisection_and_false_position.py`, `methods/04_newton_raphson.py`, `methods/05_bairstow_method.py` printed box-drawing/math Unicode (─═→▶≈×√±∛Δε₀⁴⁵Ω∞ etc., including some stored as `\uXXXX` literal escapes that a plain byte-scan misses but still evaluate to real Unicode at print time). Confirmed by actually running these files under this machine's real default console codepage (cp1252, confirmed via `chcp`) — 3 of the 4 crashed with `UnicodeEncodeError` before the fix. All Unicode replaced with plain ASCII equivalents (─→-, ═→=, ≈→~, ×→x, √→sqrt, Δ→d, ε→e, ₀→0, etc.). Re-ran all 4 files after fix — all exit code 0 now, no wrapper needed since there's no Unicode left to encode.
`basic/01`, `basic/07`, `prev_year_solves/06` already had a UTF-8 wrapper and were left alone (already safe). `basic/02` and all of `prev-section/*` were already safe (Unicode only inside docstrings or matplotlib label strings — never reaches console `print()`, matplotlib renders those itself and isn't affected by console codepage).

## `ea` formula — RESOLVED & RESTORED FOR TESTING
User requested to restore the practical small-number epsilon guards (`abs(x) < 1e-12` or `1e-15`) with absolute difference fallback (`abs(diff) * 100`) while preserving the exact-zero `float('inf')` sentinel as an inline commented line for testing.

Restored epsilon guards across 8 affected files:
- `cheatsheets/00_exam_cheatsheet.py`: `bisection`, `false_position`, `false_position_illinois`, `newton_raphson` — restored `abs(xr) < 1e-12` / `abs(xi1) < 1e-12` guard with `abs(diff) * 100.0` fallback and commented `# elif xr == 0: ea = float('inf')`.
- `methods/03_bisection_and_false_position.py`: same update across three bracketing functions.
- `methods/04_newton_raphson.py`: same update across `_newton_raphson_core` and `advanced_newton_raphson`.
- `prev-section/a1-diode_equation.py`, `b1-multi_root_scanner.py`, `c2-first_match_scanner.py`, `b2-pathological_functions.py`: restored `abs(xr) >= 1e-15` guard with `abs(diff) * 100.0` fallback and commented `# ea = float('inf')`.
- `prev-section/c1-log_scale_convergence.py`: updated `solve_bisection` and `solve_false_position` with `abs(xr) >= 1e-15` guard and commented `# ea = float('inf')`.
- Re-verified all 8 touched files via execution tests — all exit code 0, cleanly handles both standard non-zero convergence and zero / subnormal edge cases without `ZeroDivisionError`.

**Not touched — no slide ground truth provided for it:**
`05_bairstow_method.py` (and the Bairstow section of `cheatsheets/00`) has its own `ea_r`/`ea_s`
guard (`if r!=0 else 1e9` / `else 1e9`, cheatsheet uses `1e9` too, `methods/05` uses `1e10`) — the
`rootfinding-reference.md` file explicitly states Bairstow's method is out of scope until its own
slide deck is provided, so this was left as-is. Revisit once a Bairstow slide extraction exists.

`prev_year_solves/06_prev_batch_all.py` — still has no guard at all (would raise
`ZeroDivisionError` on an exact-zero root). Left untouched — user said stop touching this folder
for now.

Header spelling: confirmed all touched files already use the `f_prime(x_i)` ASCII spelling (not
the Unicode prime `′`), consistent with what's required — no action needed unless this changes.

## Calling-convention unification + B2-style coverage — RESOLVED this session

User's ask: make `methods/`, `basic/`, `cheatsheets/` the canonical "main algo" files to paste
functions out of during the exam, verified consistent with `prev-section/`'s (validated) behavior,
generic enough to drop-replace for whatever question type shows up (per `res/questions.md`
patterns). Two decisions were confirmed with the user before touching anything:

1. **Calling convention: global `f`/`df`, not parameter-passing.** `cheatsheets/00` and every
   `prev-section/*` file already redefine a module-level `def f(x): ...` (and `df(x)` for NR) at
   the top of the script and have every solver read that global — no `f` argument. `methods/03`
   and `methods/04` previously diverged: their `bisection`/`false_position`/`newton_raphson` etc.
   took `f` (and `df`) as explicit parameters instead (matching the reference-doc pseudocode, but
   NOT matching the rest of this codebase). User chose to make **global-redefinition** the one
   true convention everywhere. Refactored:
   - `methods/03_bisection_and_false_position.py`: `bisection`, `false_position`,
     `false_position_illinois`, `find_all_roots`, `compare_methods` now take no `f` param — they
     read the module-level `f(x)` (redefined before each `__main__` example, exactly like
     `cheatsheets/00`'s "swap f(x) and run" pattern). `find_sign_change_intervals` and
     `plot_root_finding` deliberately kept an **explicit** `fn`/vectorized-function parameter —
     they're utility/plotting helpers, not "the algorithm", matching `cheatsheets/00`'s own split
     (its `plot_standard`/`plot_logscale`/`plot_with_root` already take `f_np` explicitly rather
     than reading the global).
   - `methods/04_newton_raphson.py`: same treatment for `newton_raphson`,
     `advanced_newton_raphson`. Added a private `_newton_raphson_core(f_, df_, x0, tol, max_iter,
     verbose)` so `newton_raphson()` (reads global `f`/`df`) and `newton_raphson_numeric()` (builds
     a central-difference `df` from global `f`) share one implementation without duplicating the
     loop. `plot_newton_raphson` now takes an explicit `f_np` (vectorized, for the curve only) and
     internally calls the global-reading `newton_raphson()` — no more risk of the plot's curve and
     the algorithm's root silently using two different functions.
   - While refactoring, found and fixed a **pre-existing latent bug** in `methods/04`'s NR loop:
     if `f'(x0)` is flat on the very first iteration, the closing summary `print` referenced `ea`/
     `i` before either was ever assigned (`NameError`). Fixed by reading `history[-1]` instead, and
     printing a graceful message when `history` is empty.
   - `methods/05_bairstow_method.py` needed **no change** — it already takes `coeffs` (the
     polynomial itself) as an explicit argument in both `cheatsheets/00` and `methods/05`, which
     was already consistent (Bairstow doesn't operate on a scalar global `f(x)`, so this is the
     correct exception, not a violation of the convention).
   - Verified by running all 4 touched/added files — exit code 0, spot-checked NR residuals,
     quadratic convergence on `x^2-2`, and the oscillation/divergence failure demos (which reassign
     the global `f`/`df` mid-file via `global f, df` before calling the shared solvers).

2. **Added generic B2-style scan/verdict machinery** (was previously hardcoded only inside
   `prev-section/b2-pathological_functions.py` for one specific function). Per
   `numerical-methods-rootfinding-reference.md` Sections 6-8, added to **both**
   `basic/scanner.py` and `cheatsheets/00_exam_cheatsheet.py` (and equivalents inline in
   `methods/03`, since every file in this codebase is meant to be self-contained / copy-pasteable
   on its own — intentionally duplicated rather than cross-imported):
   - `safe_eval(f, x)` — try/except wrapper catching `ZeroDivisionError`/`ValueError`/
     `OverflowError`, returns `(value, None)` or `(None, error_str)`.
   - `coarse_scan(f, a, b, step=0.1, direct_tol=1e-8)` → `(direct_candidates, interval_candidates)`
     — classifies scan points into "already ~0" direct hits vs. sign-change bracket candidates in
     one pass, skipping undefined grid points instead of crashing.
   - `classify_and_verify(f, candidate, kind, extra_filter=None, residual_tol=1e-6)` → verdict dict
     (`type`, `x`, `f_x`, `error`, `accepted`) — residual check plus an optional caller-supplied
     extra filter (e.g. the B2 exam's literal `|xr| <= 1e-6`).
   - `methods/03`'s and `cheatsheets/00`'s `bisection`/`false_position`/`false_position_illinois`
     now also gracefully terminate (`return None, ...`) if `f` raises mid-iteration (e.g. hitting
     an asymptote), instead of crashing the whole run — matches
     `prev-section/b2-pathological_functions.py`'s proven "terminate that run" behavior, but now
     generic for any `f`, not hardcoded to the one pathological function.
   - Verified `methods/03`'s new Example 4 (the same rational pathological function as B2)
     reproduces the exact same verdicts as `prev-section/b2`: double root at x=2.5 → REJECT
     (residual ~0 but `|x|>1e-6`), asymptote interval → REJECT (blown-up residual), real root near
     x=-1.052 → REJECT too under the exam's literal `|xr|<=1e-6` filter (expected/correct per the
     spec's exact wording, not a bug — same conclusion `prev-section/b2` already documented).

## Question Set Formatting (`res/questions.md`)
Converted the raw Banglish exam question notes (Sections A1, B1, B2, C1, C2) into clean, fully structured Markdown format in `res/questions.md` (and deleted `res/questions.txt`):
- **Standardized Mathematics:** LaTeX formulas for diode characteristic equation (A1), transcendental equation (B1), rational function with asymptote (B2), cubic polynomial (C1), and logarithmic equation (C2).
- **Clear Task Specifications:** Formatted domain intervals, step sizes ($\Delta x = 0.1$), stopping criteria ($|\varepsilon_a| \le 0.0001\%$), and required tabular outputs.
- **Formalized Conceptual Questions:** Translated and structured the raw Banglish theoretical exam questions (B1 single-interval failure, B2 pathological root/asymptote behaviors, C2 log-scale bisection vs. false position convergence and endpoint stagnation).

## Per-slide study guides added this session — `codes/docs/SLIDE1_ERRORS_APPROX_STUDY_GUIDE.md` & `codes/docs/SLIDE2_ROOTFINDING_STUDY_GUIDE.md`

User's ask: one dedicated study-guide markdown file per slide deck (the two `references/*.md` files ARE the "2 slides" — errors/approximations, and root-finding), each one mapping every theory section to its exact code counterpart (file/function/line), explicitly flagging theory that has no code counterpart yet, and closing with a "how to study this code properly" section. Explicitly told not to touch the pre-existing `docs/*.md` files (`STUDY_GUIDE.md`, `NEWTON_RAPHSON_DEEP_DIVE.md`, `BAIRSTOW_DEEP_DIVE.md`) — new files only, added alongside them.

**`SLIDE1_ERRORS_APPROX_STUDY_GUIDE.md`** maps `references/numerical-methods-errors-reference.md` §1–§9 to `codes/basic/01_errors_and_approximations.py` (error formulas, Scarborough tolerance, machine epsilon, round-off demo, Taylor-series truncation), `codes/basic/07_error_tradeoff_plotter.py` (round-off/truncation V-curve), and `codes/basic/02_visualization_and_plotting.py` (plot templates). Flags two real gaps where the reference gives a formula/function with **no code implementation anywhere in `codes/`**: §3.4's significant-digit utilities (`sci_exponent`/`mth_sig_digit_place`) and §6.3's left-Riemann-sum integration truncation demo (`left_riemann`) — told the user to write both themselves from the reference pseudocode as practice, since there's nothing to copy for either. Also documents where the actual code takes a shortcut vs. the reference (e.g. `taylor_exp` in `basic/01` is a fixed-n, true-error-only specialization of the reference's generic Ea/εa-table-building `maclaurin_series` callback pattern).

**`SLIDE2_ROOTFINDING_STUDY_GUIDE.md`** maps `references/numerical-methods-rootfinding-reference.md` §1–§13 to `codes/methods/03_bisection_and_false_position.py`, `codes/methods/04_newton_raphson.py`, and `codes/basic/scanner.py` (coarse_scan/safe_eval/classify_and_verify — cross-referenced against the near-identical copies duplicated in `methods/03` and `cheatsheets/00`, explained as a deliberate self-contained-per-file design choice, not accidental drift). Explicitly scopes out Bairstow — reiterates the reference doc's own §13 "out of scope, no slide deck yet" note and tells the user `methods/05_bairstow_method.py` is NOT covered by this guide even though it exists and works. Flags the Illinois false-position method (`false_position_illinois` in `methods/03`/`cheatsheets/00`) as a bonus beyond this slide's reference doc — the reference only describes FP stagnation as a drawback to explain, it never mentions "Illinois" by name; that fix comes from the broader `docs/STUDY_GUIDE.md`, not this slide. Documents the `ea = 100.0` iteration-1 sentinel (and the `abs(xr) < 1e-12` epsilon guard, with the reference's literal `float('inf')` approach kept as a commented-out alternative line) as an intentional divergence from the reference's "print blank on iteration 1" convention, and tells the user which answer to give depending on whether a question is asking about the theory or about their own code's behavior.

Both files end with a numbered "How to Study This Code Properly" section: derive the formula by hand from the reference before looking at code, run every file (don't just read), dry-run 2-3 iterations of a method by hand and diff against a `verbose=True` run, deliberately trigger every documented failure mode once under low stakes, and use `prev-section/*.py` as an answer key to check against *after* attempting a `res/questions.md` item cold — not as the first thing to read.

