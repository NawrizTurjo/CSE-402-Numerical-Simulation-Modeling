# Study Guide — Slide 2: Root Finding (Bisection, False-Position, Newton-Raphson)

**Source slide:** "Root Finding: Bisection, False-Position & Newton-Raphson Method" (Nafis Tahmid, CSE, BUET)
**Ground-truth theory doc:** [`references/numerical-methods-rootfinding-reference.md`](../../references/numerical-methods-rootfinding-reference.md) — read that file's section number whenever this guide says "Ref §N".
**Your starting point (code):** `codes/methods/03_bisection_and_false_position.py`, `codes/methods/04_newton_raphson.py`, `codes/basic/scanner.py`. Secondary/cross-check copies of the same logic: `codes/cheatsheets/00_exam_cheatsheet.py` (all-in-one, condensed) and `codes/prev-section/*.py` (worked exam-style answers).

**Scope note, carried over from the reference doc itself (Ref §13):** Bairstow's method is *not* part of this slide deck — no ground-truth slide content for it exists yet. `codes/methods/05_bairstow_method.py` and the Bairstow section of `cheatsheets/00` do exist and work, but they were built without a slide to check against, so **this study guide does not cover them** — don't treat their presence in the codebase as evidence they're examinable from *this* slide. If Bairstow is separately confirmed in scope, it needs its own guide once its slide deck is available.

---

## 1. Root-Finding Fundamentals & the Sign-Change Theorem (Ref §1) → `basic/scanner.py` + conceptual

Ref §1.2's Intermediate Value Theorem nuances (sign-change is sufficient-not-exhaustive; a root that merely touches zero produces no sign change at all) are pure theory — no function "implements" a theorem. What the code gives you is the *practical consequence* of §1.3 ("never run bisection over one wide interval — scan first"):

```python
def find_sign_change_intervals(f, start, end, step=0.1):   # scanner.py lines 22-56
def coarse_scan(f, a, b, step=0.1, direct_tol=1e-8):        # scanner.py lines 71-96
```

`coarse_scan` is the fuller version — it returns **two** disjoint candidate lists (Ref §6): `direct_candidates` (points where `|f(x)|` is already ≈0) and `interval_candidates` (bracketing sign-change pairs). `find_sign_change_intervals` is the simpler variant that only returns interval brackets. Both exist because different exam question styles ask for different things — a plain multi-root scan (Ref §1.3 style) only needs the simple version; a pathological/discontinuous-function question (Ref §7–8 style, e.g. exam B2) needs `coarse_scan`'s direct/interval split.

**The exact same two functions are duplicated three more times** — inside `methods/03_bisection_and_false_position.py` (lines 58–95, 352–362), `cheatsheets/00_exam_cheatsheet.py` (lines 47–56, 73–90), and `basic/scanner.py` itself. This isn't accidental drift — it's a deliberate exam-day design choice (see `AGENT.md`'s "calling-convention unification" notes): every file is meant to be **self-contained and copy-pasteable on its own** during a timed exam, so cross-file imports were avoided on purpose. When you study this, don't ask "why is this copied four times" — ask "do all four copies actually agree with each other," which they do (verified by the AGENT.md audit). Pick whichever file you're already using as your exam template and use *its* copy; don't mix-and-match across files.

**The 1/x singularity example from Ref §3.4** (`f(xl)·f(xu) < 0` satisfied purely by a discontinuity, not a real root) is why `coarse_scan`/`safe_eval` exist at all — trace `safe_eval` (scanner.py lines 59–68) and notice it doesn't try to detect singularities directly; it just catches the *crash* (`ZeroDivisionError`/`ValueError`/`OverflowError`) if evaluating `f` at a scan point fails, and treats that grid point as "skip, not fatal." The actual singularity-vs-root distinction is caught later, at verification time (§8 below), not at scan time.

---

## 2. Stopping Criteria (Ref §2, condensed restatement of Slide 1 §2) → every solver's inner loop

Same `εa = |(x_new − x_old)/x_new| × 100` formula as Slide 1, just now applied to root estimates. You'll find the **identical three-branch pattern** repeated in every solver in `methods/03` and `methods/04`:

```python
if xr_old is None:
    ea = 100.0
elif abs(xr) < 1e-12:
    # ea = float('inf')  # Alternative exact-zero infinity sentinel
    ea = abs(xr - xr_old) * 100.0
else:
    ea = abs((xr - xr_old) / xr) * 100.0
```
(e.g. `methods/03` lines 156–162 inside `bisection()`, lines 233–239 inside `false_position()`; `methods/04` lines 91–95 inside `_newton_raphson_core`.)

**This is a real, deliberate divergence from the reference doc worth understanding, not a bug.** Ref §2.1's code pattern (and Ref §1.4 back in Slide 1) says: print εa as blank/`—` on iteration 1, don't invent a number. The actual code instead sets `ea = 100.0` as a sentinel on iteration 1 (so the loop's `if ea <= tol` check simply never fires that early — 100% is always above any realistic tolerance). Functionally the two approaches behave the same in practice, but if you're asked "what happens to εa on the first iteration" as a written question, the *theoretically correct* answer is Ref §1.4/§2.1's ("undefined, not printed as a number"), and the code's `100.0` placeholder is an engineering convenience for the loop condition — know which one to state depending on whether you're answering theory or explaining your code's behavior.

The `abs(xr) < 1e-12` branch is **not in the reference doc's code pattern at all** — it was added to the practice codebase specifically to avoid `ZeroDivisionError` when a root estimate lands exactly (or very near) zero. The commented-out `# ea = float('inf')` line right above it is the reference doc's actual approach (Ref §2.1: `xr == 0 → undefined`), kept as a comment for reference/testing rather than deleted. If an exam question's expected answer explicitly wants the `float('inf')` sentinel behavior instead of the practical epsilon-guard fallback, that's a one-line swap — know both exist and why.

---

## 3. Bisection Method (Ref §3) → `methods/03_bisection_and_false_position.py`, `bisection()`, lines 118–187

| Theory (Ref §3.1 algorithm steps) | Code |
|---|---|
| Step 1: confirm `f(xl)·f(xu) < 0` | lines 126–134 (raises `ValueError` if not satisfied, or if `f` is undefined at either endpoint) |
| Step 2: `xm = (xl+xu)/2` | line 147, comment `# <- BISECTION FORMULA: midpoint` |
| Step 3: sign test decides new bracket | lines 176–179: `if fxl * fxr < 0: xu = xr else: xl, fxl = xr, fxr` |
| Step 4/5: εa against previous xm, repeat until `≤ εs` | lines 156–172 |

**Read the full docstring (lines 119–124) before the code** — it explains a behavior the reference doc's simpler code pattern (Ref §3.2) doesn't have: this version **terminates gracefully** (`return None, history`) if `f` becomes undefined mid-run (line 148–153), instead of crashing. That's the direct code answer to Ref §7's "runtime domain errors mid-iteration" concern (an asymptote inside the bracket) — trace it against `prev-section/b2-pathological_functions.py` to see it exercised on a real pathological function (`(x-2.5)²(x+1.052)/(x-3.551)` on `[-2,5]`, asymptote at `x=3.551`).

**Advantages/Drawbacks (Ref §3.3/§3.4) are not "in" the code as comments** — they're things you verify by *running* the code on the right examples: the 1/x singularity drawback (Ref §3.4) is demonstrated conceptually, not literally coded as a named example anywhere in `methods/03`; if asked to demonstrate it, you'd write `f = lambda x: 1/x` yourself, bracket `[-2, 3]`, run `bisection`, and observe it "converges" to something near 0 with a blown-up residual — then explain *why* that's not a real root (Ref §7, §8 below).

---

## 4. False-Position Method (Ref §4) → `methods/03`, `false_position()` (lines 194–267) + `false_position_illinois()` (lines 274–345)

| Theory | Code |
|---|---|
| §4.1 secant formula `xr = [xu·f(xl) − xl·f(xu)] / [f(xl) − f(xu)]` | `false_position()` line 224: `xr = xu - fu * (xl - xu) / (fl - fu)` — this is Ref §4.1's *second* "increment" form, algebraically identical, not the first form written in the reference's headline equation. Confirm on paper they're the same before assuming a typo. |
| §4.2 bracket update, identical logic to bisection | lines 252–255 |
| §4.4 endpoint-stall instrumentation (`xl_updates`/`xu_updates`) | lines 211–212 (counters), 253/255 (`xu_updates += 1` / `xl_updates += 1`), printed warning at lines 262–265 if either count stays 0 |

**§4.4 is the section most worth tracing carefully** — the reference doc frames it as a *separate* function (`false_position_instrumented`), but the actual code folds the instrumentation directly into the main `false_position()` (it always counts updates, no separate variant needed). If asked to report "how many times did xl vs xu update," you don't need a special function — the numbers are already tracked and printed by the default call.

**Illinois modification (`false_position_illinois`, lines 274–345) is a genuine bonus beyond this slide's reference doc** — search the reference file and you won't find "Illinois" mentioned anywhere in it; the reference's §4.5 only describes the *stagnation problem* conceptually ("f strongly curved near one endpoint → that endpoint never updates → error shrinks slowly") without giving a fix. The Illinois fix (halving the stagnant endpoint's `f`-value to force the secant to switch sides — lines 330–337) comes from `docs/STUDY_GUIDE.md`'s broader treatment, not from this slide's ground truth. **Know the distinction:** the slide/reference expects you to *explain* stagnation as a drawback (why FP isn't guaranteed faster than bisection — Ref §4.5, and the ln(x) benchmark: bisection 34 iters vs FP 126 iters), not necessarily to *fix* it with Illinois — but the fix is there in code if a question does ask for it.

---

## 5. Newton-Raphson Method (Ref §5) → `methods/04_newton_raphson.py`

| Theory | Code |
|---|---|
| §5.1 formula `xi+1 = xi − f(xi)/f'(xi)` | `_newton_raphson_core()`, line 88 |
| §5.2 flat-tangent check *before* dividing | lines 82–85: `if abs(dfxi) < 1e-12: ... break` |
| §5.3 iteration table columns `Iter\|Vi\|f(Vi)\|f'(Vi)\|Vi+1\|εa%` | printed header, lines 68–73 |

**Architecture note that matters for exam-day copy-pasting:** unlike the reference doc's single `newton_raphson(f, fprime, x0, ...)` function, the actual code has **three layers**:
- `_newton_raphson_core(f_, df_, x0, tol, max_iter, verbose)` (lines 67–117) — the shared loop, takes `f`/`df` as explicit parameters.
- `newton_raphson(x0, ...)` (lines 120–122) — the one you actually call; reads the module-level global `f`/`df` and forwards to the core.
- `newton_raphson_numeric(x0, h=1e-6, ...)` (lines 125–132) — same idea, but builds `df` itself via central difference `(f(x+h)-f(x-h))/(2h)` when you don't have (or don't want to derive) `f'(x)` analytically.

This split exists purely so `newton_raphson()` and `newton_raphson_numeric()` don't duplicate the loop (see `AGENT.md`'s refactor notes) — functionally, calling `newton_raphson(x0)` after redefining global `f`/`df` gives you exactly Ref §5.3's algorithm. Don't call `_newton_raphson_core` directly in an exam answer; it's a private implementation detail (leading underscore).

### 5.5 Drawbacks — each one has a named, runnable demo (not just a description)

| Ref §5.5 failure mode | Code demo |
|---|---|
| 1. Divergence near inflection points | described in reference only; no dedicated function in `methods/04` — the `(x-1)³+0.512` example lives in the reference doc's prose, not as code here |
| 2. Division by (near) zero | `_newton_raphson_core`'s guard (lines 82–85) is the *defense*; `demonstrate_cube_root_divergence()` (lines 267–292) is the closest runnable analog, using `cbrt(x)` where the derivative blows up (not vanishes) near the root — read carefully, it's the *opposite* numerical hazard (infinite derivative, not zero derivative), both breaking NR for related reasons |
| 3. Oscillation with no real root | not directly demoed (the reference's `x²+2` example isn't coded); `advanced_newton_raphson`'s cycle detector (§5.6-adjacent, see below) is the general defense against oscillation-family failures |
| 4. Root jumping | not directly demoed in `methods/04`; conceptually the same family as the oscillation cycle-detector's blind spot — cycle detection catches *repeating* jumps, not a single silent jump to a distant unintended root |

`demonstrate_oscillation_failure()` (lines 248–264) runs the reference's *other* canonical failure — `f(x)=x³−2x+2`, `x0=0`, 0→1→0→1 cycling — via `advanced_newton_raphson()` (lines 139–198), which is a **superset** of the plain `newton_raphson`: it adds a multiplicity modifier `m` (Ref §5.5 point 4's "slow multiple roots" fix, formula `x_{i+1} = x_i − m·f(x_i)/f'(x_i)`) and a visited-history cycle check (lines 180–185: compares each new `xi1` against every previously visited `xi` within `1e-6`).

**Practical implication (Ref §5.6):** every NR call you write in the exam needs (a) the flat-derivative guard, (b) a hard `max_iter` cap, (c) a residual check `f(root)` at the end. All three are already present in `_newton_raphson_core` — you get them for free by calling `newton_raphson()`/`newton_raphson_numeric()` rather than reimplementing the loop from scratch.

---

## 6. Coarse Scanning Convention (Ref §6) → covered under §1 above (`basic/scanner.py`)

Ref §6's `coarse_scan` code block is reproduced near-verbatim in four places (scanner.py, methods/03, cheatsheets/00 — see §1's note on deliberate duplication). One thing worth double-checking yourself: Ref §6 explicitly says Newton-Raphson "doesn't consume this scan the same way... a coarse scan can still inform a reasonable x0." None of the NR example blocks in `methods/04`'s `__main__` (lines 299–370) actually call `coarse_scan`/`find_sign_change_intervals` to *pick* their `x0` — each example just states `x0` directly from domain reasoning (e.g. "physical height h∈[0,8], f(0)>0, f(1)<0, root in [0,1], pick h0=0.5"). If a question wants you to justify an NR starting guess via scanning, you're combining scanner.py's function with methods/04's solver yourself — they aren't pre-wired together in the example code.

---

## 7. Undefined-Value / Singularity Handling (Ref §7) → `safe_eval` everywhere + graceful termination in every solver

Two distinct failure modes per Ref §7:
1. **Sign change caused by a singularity** (1/x-style) — defended against by the *acceptance filter* in §8 below, not by the scan/bracket code itself (which can't tell a singularity from a root just by seeing a sign flip — that's the whole point of Ref §7 point 1).
2. **Runtime domain errors mid-iteration** — defended against directly: every solver (`bisection`, `false_position`, `false_position_illinois` in `methods/03`) wraps its per-iteration `f` evaluation in `safe_eval` and returns `None`/terminates that specific run rather than raising (see §3's note above on `bisection`'s graceful-exit branch, lines 148–153).

`safe_eval(f, x)` itself (`scanner.py` lines 59–68, duplicated in `methods/03` lines 58–67 and `cheatsheets/00` lines 64–71) is deliberately small — three exception types caught (`ZeroDivisionError, ValueError, OverflowError`), nothing fancier. Know this list; if a question's pathological function raises something else (e.g. a custom exception), this exact `except` clause won't catch it and you'd need to extend it.

---

## 8. Acceptance Filtering & Root Reporting (Ref §8) → `classify_and_verify()`

```python
def classify_and_verify(f, candidate, kind, extra_filter=None, residual_tol=1e-6):
    fx, err = safe_eval(f, candidate)
    accepted = fx is not None and abs(fx) <= residual_tol
    if accepted and extra_filter is not None:
        accepted = extra_filter(candidate)
    return {'type': kind, 'x': candidate, 'f_x': fx, 'error': err, 'accepted': accepted}
```
(scanner.py lines 99–112; near-identical copy in `methods/03` lines 98–111.)

This is **the** function that resolves Ref §7's singularity-vs-root ambiguity: it doesn't matter that bisection "converged" on something — `classify_and_verify` re-evaluates `f` at the final estimate and only accepts it if the residual is genuinely small. `type` (`'direct'` vs `'interval'`) matches Ref §8's reporting requirement directly. `extra_filter` is the hook for **problem-specific secondary filters** Ref §8 calls out explicitly (its own example: "one sample problem required `|xr| ≤ 1e-6`") — study `prev-section/b2-pathological_functions.py` to see `extra_filter=lambda x: abs(x) <= 1e-6` applied literally, and notice (per that file's own documented findings) that **most candidates end up REJECTed under that filter** — that's the exam's literal wording working as intended, not a bug to "fix."

**Full worked trace worth doing yourself once:** run `methods/03`'s `__main__` Example 4 (lines 522–551) — pathological function `(x-2.5)²(x+1.052)/(x-3.551)` on `[-2,5]` — and match its printed verdict table against `prev-section/b2-pathological_functions.py`'s output. AGENT.md confirms these two are meant to (and do) reproduce identical verdicts: double root at 2.5 → REJECT, asymptote interval → REJECT (blown-up residual), real root near −1.052 → REJECT too under the strict `|xr|≤1e-6` filter. If your two runs disagree, something in your copy has drifted from the verified version.

---

## 9. Method Comparison Deliverables (Ref §9) → `methods/03`, `compare_methods()`, lines 392–412

Runs bisection, false-position, and Illinois FP on the same bracket and prints one shared table (root, iteration count, final `f(xr)`, final εa, plus the FP stagnation counts). This directly satisfies Ref §9's "one comparison table, not two separate reports" requirement — study the `__main__` Example 1 (ln(x) on `[1e-4, 1e4]`, lines 453–466) to see it reproduce the reference's own documented benchmark (bisection 34 iterations vs FP's stagnation-heavy 126).

---

## 10. Plotting Conventions (Ref §10) → same templates as Slide 1 §7, extended

Nothing new mechanically — `plot_root_finding()` in `methods/03` (lines 421–446) and `plot_newton_raphson()` in `methods/04` (lines 208–241) are root-finding-specific variants of Slide 1's `basic/02` templates. The one new visual idea `plot_newton_raphson` adds beyond Slide 1's templates: it draws the **actual tangent lines** for the first 6 iterations (lines 222–230), which is the most direct visual proof you can put on an exam answer sheet that NR really is "follow the tangent to the x-axis, repeat" (Ref §5.1) rather than an abstract formula.

---

## 11. Theoretical Short-Answer Prompt Bank (Ref §11) — no code, memorize + explain

These four recurring prompt patterns are answered by *reasoning about* the code/theory above, not by running anything new:
- Why no-sign-change doesn't guarantee no root → Ref §1.2 + the touching-root example (x² at x=0).
- Why residual-checking is still needed after convergence → Ref §3.4/§7/§8, and `classify_and_verify`'s existence *is* your code evidence for this answer.
- Why FP's faster convergence isn't guaranteed → Ref §4.5, and `compare_methods()`'s ln(x) benchmark *is* your code evidence.
- Why scan-then-refine beats one-shot bisection over a wide domain → Ref §1.3, and the multi-root Section-B example (`methods/03` Example 2, lines 480–497) *is* your code evidence (misses roots if you don't scan first).

**Exam-answer pattern:** for each of these, cite the *mechanism* (as this guide's tables point you to) plus, if useful, the specific file/line or example that demonstrates it — "explain the mechanism," per Ref §11's own framing, not "restate the definition."

---

## 12–13. Formula Sheet & Out-of-Scope (Ref §12–13)

Ref §12's table is your final flashcard pass — every row is already cited above. Ref §13 reiterates: Bairstow, secant method, Brent's method, trapezoidal/Simpson's integration are **out of scope for this slide** regardless of what exists in `methods/05_bairstow_method.py` or elsewhere in the codebase. If you see Bairstow content while browsing `codes/`, remember it wasn't built from a checked slide deck (see the top of this guide).

---

## How to Study This Code Properly

1. **Study `methods/03` and `methods/04` as the primary texts; treat `cheatsheets/00` as the exam-day condensed cheat-sheet, not a separate thing to learn from scratch.** They implement the same algorithms with the same safety guards — `cheatsheets/00`'s versions are just flattened (no `_core` helper split, slightly shorter docstrings) for fast copy-pasting under time pressure. Learn the concepts from `methods/03`/`04`'s fuller docstrings and comments; use `cheatsheets/00` only to rehearse the "find function, copy, adapt f(x), run" motion you'll actually do in the exam.

2. **For each method, do a full dry-run by hand on paper first, then check against a `verbose=True` run.** Pick a simple bracket (e.g. `f(x)=x³−x−1` on `[1,2]`, already the default in both files), compute 2–3 iterations of bisection by hand, then run `bisection(1, 2, tol=0.0001)` and compare your hand-computed `xr`/`ea` values row by row against the printed table. This is the fastest way to catch a formula misunderstanding before the exam does it for you.

3. **Deliberately trigger every failure mode once.** Run `demonstrate_oscillation_failure()` and `demonstrate_cube_root_divergence()` in `methods/04` as-is, then modify `bisection`'s bracket to `f=lambda x: 1/x, xl=-2, xu=3` yourself and watch it "converge" to garbage near the singularity — then explain in your own words (out loud or written) why the residual check catches it. Failure modes you've triggered yourself under low stakes are failure modes you'll recognize instantly under exam stress.

4. **Internalize the "redefine the global `f`/`df`, then call the solver with no `f` argument" convention** — it's used consistently across `methods/03`, `methods/04`, `cheatsheets/00`, and every `prev-section/*.py` file. This is *the* thing that makes copy-paste-adapt fast on exam day: you never touch the solver's internals, you only ever redefine `f(x)` (and `df(x)` for NR) at the top and re-run. Confirm you understand *why* this convention was chosen over passing `f` as a parameter (see `AGENT.md`'s unification notes) — because every file needs to be self-contained and quickly re-targetable at a new question without touching function signatures.

5. **Use `prev-section/*.py` as your "was I right" answer key**, not as your first read. Once you can explain a method from `methods/03`/`04`, go solve the matching `res/questions.md` item cold, *then* diff your approach against the matching `prev-section/` file (remember the AGENT.md note: filenames there don't line up 1:1 with exam labels — `c1-*.py` answers exam C2, `c2-*.py` answers exam C1; content is right, only names are swapped).

6. **Keep the acceptance-filtering habit (§8) active in every practice run, even when the problem seems clean.** It's tempting to skip `classify_and_verify` on a "nice" function where you're confident the root is real — but the exam explicitly tests whether you apply the residual check *unconditionally*, not just when you suspect a trick. Make it muscle memory now so you don't have to remember to add it under time pressure.
