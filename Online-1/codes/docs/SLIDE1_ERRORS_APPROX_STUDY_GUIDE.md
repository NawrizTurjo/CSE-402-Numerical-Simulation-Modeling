# Study Guide — Slide 1: Approximations, Round-off & Truncation Errors

**Source slide:** "Introduction, Approximations & Errors" (Nafis Tahmid, CSE, BUET)
**Ground-truth theory doc:** [`references/numerical-methods-errors-reference.md`](../../references/numerical-methods-errors-reference.md) — read that file's section number whenever this guide says "Ref §N".
**Your starting point (code):** `codes/basic/01_errors_and_approximations.py`, `codes/basic/07_error_tradeoff_plotter.py`, `codes/basic/02_visualization_and_plotting.py`.

This guide does one job: for every theory concept on the slide, it tells you **exactly which function, in which file, at which lines, implements it** — so you study the code as a worked example of the theory, not as a black box. Where the slide/reference describes something the codebase does *not* yet implement, that's flagged explicitly as a gap for you to close yourself (closing it is itself good exam practice).

---

## 1. Error Definitions (Ref §1) → `basic/01_errors_and_approximations.py`

| Theory (Ref §1.x) | Formula | Code |
|---|---|---|
| §1.1 True error Et | `Et = True − Approx` | `true_error()`, lines 33–35 |
| §1.2 Relative true error εt | `εt(%) = |Et/True| × 100` | `true_relative_error_pct()`, lines 37–39 |
| §1.3 Approximate error Ea | `Ea = Present − Previous` | not returned standalone — folded directly into εa below |
| §1.4 Relative approximate error εa | `εa(%) = |Ea/Present| × 100` | `approx_relative_error_pct()`, lines 41–49 |

**Read this pairing carefully:** `approx_relative_error_pct(x_new, x_old)` (line 41) is *the* function that matters most in the whole course — every iterative method you'll write later (bisection, false-position, Newton-Raphson) is just this same formula applied to successive root estimates instead of successive series terms. If you understand these 9 lines cold, you already understand 80% of the "error" column in every iteration table you'll ever print.

**Divergence from the reference worth noticing:** Ref §1.4's "Implementation note" says εa is *undefined* on iteration 1 and should print as blank/`—`, not zero. The actual code takes a simpler shortcut: `approx_relative_error_pct` returns `float('inf')` when `x_new == 0` (line 47–48) but doesn't have a "no previous value yet" branch at all — that's handled by the *caller* in every method in `methods/03` and `methods/04` (they track `xr_old = None` and only start comparing once a previous value exists). So: this one function is the raw formula; the "blank on iteration 1" behavior lives one level up, in the loop that calls it. Don't expect to find it inside this function.

---

## 2. Stopping / Convergence Criteria (Ref §2) → `basic/01`, lines 51–58

```python
def scarborough_tolerance(n_sig_figs):
    return 0.5 * 10 ** (2 - n_sig_figs)
```

This is Ref §2.2's significant-digits criterion (`|εa(%)| ≤ 0.5×10^(2−m)`), coded directly. Ref §2.1's *other* form — a flat pre-specified `εs` like `0.0001%` — has no dedicated function anywhere, because it doesn't need one: it's just `abs(eps_a_percent) <= tol` inlined straight into every solver's loop condition. You'll see that exact inline pattern later in `methods/03_bisection_and_false_position.py` and `methods/04_newton_raphson.py` (e.g. `if xr_old is not None and ea <= tol: break`) — same criterion, no separate helper function needed because it's a one-line comparison.

**Study tip:** run `basic/01` and look at the printed Scarborough table (Part A/B of the script). Confirm by hand that `n=2 → 0.5%` and `n=6 → 0.00005%` match Ref §2.2's worked example (`m=2 → 0.5×10⁰ = 0.5%`).

---

## 3. Significant Digits (Ref §3) → **gap, not implemented**

Ref §3.4 gives two small utilities:
```python
def sci_exponent(x): ...        # n such that x = m*10^n
def mth_sig_digit_place(x, M): ...
```
Neither exists anywhere in `codes/`. This is real, testable theory (Ref §3.1–3.3: "x̂ correct to M significant digits when `|x−x̂| ≤ 0.5×10^(n−M+1)`", and the equivalence proof between that and the εt criterion). If a written/short-answer question asks you to locate the M-th significant digit of a number, or to derive the equivalence, you're working from the reference doc directly, not from code — there's nothing to run.

**If you want a code counterpart to practice with**, write it yourself in a scratch file using exactly the two signatures above and the worked example in Ref §3.2 (`3725.6`, 3rd sig digit → tens place) as your test case. This is a good five-minute active-recall exercise, not a large task.

---

## 4. Sources of Error (Ref §4) — conceptual only, no code

Four categories: modeling error, data/programming error, round-off (§5), truncation (§6). Only the last two have code (below). The trunnion/thermal-expansion case study (Ref §4, point 1) is pure short-answer material — be ready to explain *why* a wrong model can pass internal consistency checks and still be wrong. No file to run here; this is memorization + explanation, not code-tracing.

---

## 5. Round-off Error (Ref §5) → `basic/01`, Part B & C (lines 61–113)

| Theory | Code |
|---|---|
| Machine epsilon (hardware precision floor) | `compute_machine_epsilon()`, lines 65–73 |
| §5.2 worked 1/3 rounding example | Part C block, lines 96–104 |
| Subtractive cancellation (bonus, not in slide text but same failure family) | lines 106–113: `naive = sqrt(x+1)-sqrt(x)` vs `stable = 1/(sqrt(x+1)+sqrt(x))` |

**How to read `compute_machine_epsilon`:** it's a physical demonstration of "finite precision," not an abstract formula. Trace it by hand for 2–3 loop iterations: `eps` keeps halving until `1.0 + eps/2.0 == 1.0`, i.e. until eps/2 is too small to change 1.0's stored bit pattern at all. That halt condition **is** machine epsilon's definition — the code doesn't compute it via a formula, it *discovers* it by probing float64's representable range. This is worth being able to reproduce on paper/whiteboard if asked to "explain machine epsilon."

The subtractive-cancellation block (lines 106–113) isn't explicitly named in the reference doc's Round-off section, but it's the same underlying phenomenon the reference discusses in Ref §6.2's differentiation context ("shrinking h shrinks truncation error") taken to its round-off extreme — see `basic/07` below for where this becomes the main event.

---

## 6. Truncation Error (Ref §6) — three sub-cases

### 6.1 Series truncation (Maclaurin/Taylor) → `basic/01`, Part D (lines 116–144)

```python
def taylor_exp(x, n_terms):
    total = 0.0
    term  = 1.0
    for k in range(n_terms):
        total += term
        term  *= x / (k + 1)
    return total
```

This implements the recurrence `term_{k+1} = term_k · x/(k+1)` — the *general* Maclaurin recurrence from Ref §6.1 specialized to `e^x`. Compare directly against the reference's generic `maclaurin_series(term_recurrence, x, ...)` (Ref §6.1 code block): the reference version accepts a `term_recurrence` callback so it works for *any* series (ln(1+x), sin(x), etc.), builds a full `n | Sn | Ea | |εa|% | sig-digits` table, and stops automatically once εa drops below a threshold. `taylor_exp` in the actual code is the **specialized, non-generic, fixed-iteration-count version** — it always runs exactly `n_terms` terms and only prints `n | approx | Et | εt%` (true-error columns, since `math.e` is known here), not the Ea/εa approximate-error columns the reference's generic table wants.

**Gap worth knowing about:** if an exam question hands you a *different* series (not e^x) and asks for the Ea/εa/significant-digits table format, `taylor_exp` as written won't directly give you that — you'd adapt it toward the reference's generic `maclaurin_series` shape (callback + Ea-based stopping) rather than copy `taylor_exp` verbatim. Know both shapes; know why the reference's version generalizes and this one doesn't.

Part F (lines 169–196) visualizes convergence: multiple `taylor_exp` curves at increasing `n_terms` plotted against the true `e^x` curve, showing how truncation error shrinks visually as terms are added.

### 6.2 Truncation in differentiation → `basic/01` Part E (lines 148–166) + `basic/07` (whole file)

Ref §6.2's forward-difference formula `f'(x) ≈ [f(x+h) − f(x)]/h` is inlined directly in `basic/01`'s Part E loop (line 163: `approx = (math.sin(x0 + h) - math.sin(x0)) / h`) rather than factored into a standalone `forward_diff(f, x, h)` function — the reference doc *does* give that as a separate function; the actual code just writes the one-liner inline since it's only used once here. Functionally identical, just not factored out.

**`basic/07_error_tradeoff_plotter.py` is the important extension** — it's not really "Slide 1 §6.2" in isolation, it's §5 (round-off) and §6.2 (truncation) collided into one demonstration, which is exactly the point Ref §6.2 gestures at ("shrinking h shrinks truncation error... comparing successive h's gives Ea") but doesn't fully unpack. Read this file end to end:
- `h_values = np.logspace(0, -20, 100)` — sweeps step size from `10^0` down to `10^-20`.
- For each `h`, computes forward-difference error against the *known* true derivative of `e^x` at `x=1`.
- Finds `optimal_h` at `argmin(errors)`.
- Plots the classic **V-shaped curve** (log-log) — large `h` → truncation-dominated (left arm), tiny `h` → round-off-dominated via subtractive cancellation (right arm), minimum in the middle.

This is the STUDY_GUIDE.md "V-Shape" section made concrete in runnable code — if a theory question asks "why does error increase again for very small h," the answer is subtractive cancellation (§5's mechanism) amplified by dividing by a tiny `h`, and this script is your evidence, not just an assertion.

### 6.3 Truncation in integration (Riemann sums) → **gap, not implemented**

Ref §6.3 gives `left_riemann(f, a, b, N)` and a worked example (f(x)=x² on [3,9], exact=234, N=2→135, N=4→182.25). **No file in `codes/` implements this at all.** If integration/Riemann-sum truncation shows up as a question, you are writing it from the reference formula directly — there's no existing template to copy. This is the single biggest "theory ahead of code" gap in this slide. Practice writing `left_riemann` once by hand and reproducing the N=2/N=4 numbers from the reference before the exam, so you're not deriving it cold under time pressure.

---

## 7. Visualization & Plotting Conventions (Ref §7) → `basic/02_visualization_and_plotting.py`

| Convention (Ref §7) | Code |
|---|---|
| Fine grid, `numpy.linspace(a,b,N≥500)` | `plot_function()`, lines 48–69 |
| axhline/axvline, labels, title, grid on every plot | same function, lines 58–65 |
| Log-scale for wide-magnitude domains | `plot_function_logscale()`, lines 76–97 (uses `np.logspace`, `plt.xscale('log')`) |
| Mark sign-change brackets (anticipates root-finding, slide 2) | `plot_with_brackets()`, lines 104–137 |
| Convergence curve (semilogy) | `plot_convergence()`, lines 144–161 |
| Coarse sign-change scan before bracketing | `find_sign_change_intervals()`, lines 29–41 — this is Ref §8's scanning convention, technically borrowed one section early since Slide 1's own material doesn't need root brackets yet |

This file is a **template library**, not something with "theory" of its own beyond §7's bullet points — the whole point is that every plot you make on the exam should visually match one of these four templates. Study it by *running* the `if __name__ == '__main__':` block (lines 168–201) and comparing each saved PNG against the checklist in Ref §7 (labels? grid? y=0 line? log scale used correctly for the wide-domain example?).

---

## 8. Observed Exam/Lab Conventions (Ref §8) → cross-references slide 2

Ref §8 is explicitly forward-looking ("bundles root-finding tasks... out of scope here") — it's really priming you for slide 2's conventions (`|εa| ≤ 0.0001%` default tolerance, mandatory iteration tables, scan-then-refine pattern, comparison tables, theory-explanation-alongside-code). Nothing new to trace in this file's code beyond what's already covered above; treat §8 as a bridge, not new material.

---

## 9. Quick-Reference Formula Sheet (Ref §9)

Every formula in this table already has a code citation above — use Ref §9's table as your final pre-exam flashcard pass, and for each row make sure you can point to the exact function/line without looking it up again.

---

## How to Study This Code Properly

You said this codebase is your **starting point** — treat that literally: the goal isn't to memorize these files verbatim, it's to be able to *regenerate* them under exam pressure from the theory. A concrete process that works for this material:

1. **Read the reference doc section first, formula only — no code.** Cover the code file. Try to write the function signature and body from the formula alone (e.g. from Ref §1.4's `εa = Ea/Present ×100`, write `approx_relative_error_pct` yourself). Then compare against `basic/01` lines 41–49. The gap between your version and the real one (e.g. the `x_new == 0` guard) is exactly what you'll forget under exam pressure if you skip this step.

2. **Run every file, don't just read it.** All four files in this guide (`01`, `07`, `02`, plus the reference doc) are runnable and print/plot something. Actually execute them (`python codes/basic/01_errors_and_approximations.py` etc.) and read the printed tables side by side with the reference doc's worked examples (1/3 rounding, e^1.2 Taylor series-style, e^x derivative at x=1). If your printed numbers don't match a reference worked example, that's a bug in your understanding, not a typo to shrug off.

3. **For the two flagged gaps (§3 significant-digit utilities, §6.3 Riemann sums), write them yourself once, from the reference pseudocode, before the exam** — not during it. These are exactly the kind of "theory the slide covers but the practice codebase never got around to" items that show up as a nasty surprise if a question targets them specifically.

4. **Trace the "global function, no parameters" convention now, even though it's more visible in slide 2's code.** `basic/01`'s functions all take explicit parameters (`true_error(true_val, approx_val)`, etc.) because there's no single `f(x)` being iterated on yet. Once you move to slide 2's bisection/Newton-Raphson code, the convention flips to "redefine a module-level `f(x)` and read it as a global" — noticing *why* that convention change happens (you're now iterating the *same* function many times, not computing a one-off error) will make slide 2's code far less mysterious.

5. **Keep a running "formula → 1-line code" flashcard set** as you go (Ref §9 gives you the starter list) — on exam day you're pattern-matching a question to a formula to a function shape fast, not re-deriving from first principles.
