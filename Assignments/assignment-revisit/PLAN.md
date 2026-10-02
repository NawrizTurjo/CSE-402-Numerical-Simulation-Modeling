# Assignment revisit: Gradient Methods, built from the class notes

**Roll:** 2105032 · **Topic 3:** Gradient Methods · **Deadline:** Fri 2 Oct 2026, 11:59 PM
**Sources:** `Assignment/reference/NZR Full Notes (Suchi).pdf` (class notes) and `Assignment/instructions.txt` (assignment spec)
**Rule for this folder:** the first deck in `Assignment/` stays untouched. Everything new lives in `assignment-revisit/`.

## 1. Why a second deck

The first deck went deep into convex-optimization research topics: PL inequality, Nesterov lower bounds,
Kantorovich, Barzilai-Borwein, edge of stability. That is mostly outside what this class taught.
This deck starts from the class notes and goes one honest step further, which is where the assignment's
"wow factor" marks live: proofs and analysis the class skipped, built on the same examples the class used.

## 2. What the class actually covered (from the notes)

| Notes page (date) | Content | How the deck uses it |
|---|---|---|
| p6 (16 Aug) | why GD: linear regression (weight vs height), logistic regression "squiggle", t-SNE clusters; height = intercept + slope × weight; start with intercept = 0; sum of squared residuals as the loss; the SSR-vs-intercept parabola; "GD takes big steps far from the optimum and small steps near it" | motivation section, and the running example for the whole deck |
| p7 | GD for one variable: we want slope = 0 but cannot always solve for it; step size = d(SSR)/d(intercept) × learning rate; the *schedule* (learning rate from large to small); new = old − step; stop when the step is near zero or at max steps; GD for several variables: (i) derivative for each parameter, (ii) random start (intercept 0, slope 1) | core algorithm, single- and multi-variable |
| p8 | (iii) plug in, (iv) step = slope × learning rate per parameter, (v) update, (vi) repeat until convergence; problem: millions of parameters; SGD: random subset each step, strictly one sample, in practice mini-batch; advantages (best of both, more stable in fewer steps); new data needs no restart | SGD and mini-batch section |
| p8–p11 (17 Aug) | Newton for optimization as the second-order counterpart; "why not GD?"; GD for big problems (ML), Newton for small ones; Newton can head to a maximum if started in the wrong place; if H is not positive definite, switch to GD (Levenberg-Marquardt) | GD vs Newton comparison (contrast only; Newton is topic 2) |
| p10 | Taylor's theorem: first- and second-order approximations; the function must be smooth, "otherwise this method will fail" | derivation of GD from first-order Taylor; failure modes |
| p12 (22 Aug) | stationary points: x² (min), −x² (max), x³ (saddle, f''=0); J = x₁² + x₂² with gradient (2x₁, 2x₂); "the gradient returns a vector field giving the direction and magnitude of the steepest slope"; basis of all optimization: (1) set up the objective, (2) search for zero gradient; GD arrows on J = x⁴ − 6x³ + 8x² | gradient section, stationary points, local minima |
| p12–p14 | contour plots (3-D surface as 2-D rings), the gradient perpendicular to the contour | geometry section (constrained part is topic 4, so not taught here) |
| p16 (23 Aug) | least-squares line: SSE, ∂/∂b = 0, ∂/∂m = 0, closed forms for m and b | closed-form check for the GD answers |
| p18 | multiple linear regression ŷ = w₀ + w₁x₁ + … ; MSE J = (1/2m) Σ (ŷ − y)²; update w_j ← w_j − α ∂J/∂w_j; start from initial values and iterate; the 1/(2m) has no effect on the minimiser ("scaling has no effect, shifting has") | multivariable GD, vectorised form, scaling/shifting problem |

The Quiz formula sheet (`Quiz/notes/parts/formulas_05_optimization.tex`) uses the same notation: SSR,
∂SSR/∂b = −2 Σ (yᵢ − b − mxᵢ), w_j ← w_j − α ∂J/∂w_j. The deck keeps that notation.

## 3. What the assignment requires, and how this deck meets it

| Requirement (`instructions.txt`) | Plan |
|---|---|
| Topic 3 = Gradient Methods (032 mod 15 + 1) | title slide says "Topic 3: Gradient Methods"; no Newton/constrained teaching beyond one comparison slide |
| LaTeX Beamer only | Beamer with the user's `paper-navy-beamer` theme (the look of the first deck) |
| Beyond-lecture "wow factor" | each extension sits right next to the class idea it extends, tagged with a small **Beyond class** label (list in §4) |
| Error and convergence analysis: order, stability, complexity, failure modes | learning-rate stability window, linear vs quadratic order, iteration counts, cost per iteration, failure gallery |
| 5 mathematical + 5 application problems, each solved right after | §6; solutions with no `\pause` so the PDF reads cleanly |
| `2105032.pdf` + `2105032_LaTeX_Source.zip`, source must compile | built in `assignment-revisit/`, packaged by `build.ps1` with the clean-room compile test |

## 4. Beyond-class extensions (the "wow factor", kept within reach)

1. **Why minus the gradient is the steepest way down.** The directional derivative ∇f·d plus Cauchy-Schwarz, in a few lines. It turns the note's "steepest slope" sentence into a proof.
2. **GD from Taylor.** The note derives Newton from the second-order Taylor model. The same move with the first-order model gives GD. One slide links the two methods: first order = GD, second order = Newton.
3. **The learning-rate stability window.** For f(x) = (x − 3)², the error obeys e_{k+1} = (1 − 2α) e_k. That splits α into four regimes: slow, best, oscillating and divergent. It explains why the class's "schedule" matters, and gives the general rule α < 2/f''.
4. **Order of convergence.** GD is linear (a constant ratio |1 − αf''|), while Newton is quadratic ("digits double", as the note says). This gives iteration counts for 10⁻⁶ accuracy.
5. **Contours, scaling and the zig-zag.** On J = x₁² + 4x₂² the contours are ellipses and GD zig-zags. Centring or scaling the features makes the contours round again. This is why feature scaling matters in ML.
6. **Failure gallery.** A step that is too big, a local minimum (x⁴ − 6x³ + 8x² from the notes), the x³ saddle, a flat plateau, and a non-smooth |x| (the note's "must be smooth").
7. **Computational view.** A vectorised NumPy update `w -= α * X.T @ (X @ w − y) / m`. Cost O(mn) per step against O(n³) for the normal equations or Newton, which answers the note's question "when GD, when Newton?" with numbers.
8. **SGD noise and mini-batches.** A noisy path, variance falling as 1/B, and a decaying learning rate (the note's "schedule") to settle at the end.
9. *(Optional if time allows)* **Steepest ascent with the best step**, the textbook (Chapra) variant that picks h by a 1-D search. One slide.

Out of scope for this deck: PL condition, Nesterov, momentum theory, Kantorovich, Barzilai-Borwein, Adam, edge of stability.

## 5. Deck outline (target about 50 frames, roughly 24 theory + 26 problems)

**Front:** title · assigned topic · roadmap

**Part A: the ideas**
1. *Why we need GD* (p6): fitting a line to weight-height data; SSR; why not solve slope = 0 directly. (3 frames)
2. *The gradient* (p12): derivative to gradient vector; the vector field picture; steepest-descent proof ★; stationary points (min, max, saddle). (4)
3. *The algorithm* (p7–p8): one variable on the class example (intercept only, with numbers); several variables, steps (i)–(vi); stopping rules; Taylor derivation ★. (5)
4. *Learning rate* (p7 "schedule"): the four regimes picture ★; stability window α < 2/f'' ★; schedules. (3)
5. *Convergence and cost* ★: linear order and iteration count; GD vs Newton table (order, cost per step, when to use, Newton's wrong-direction issue from p8). (3)
6. *Geometry* (p12–p14): contours and the gradient ⟂ contour; elongated contours, zig-zag, feature scaling ★. (2)
7. *Failure modes* ★: the gallery from §4.6. (2)
8. *SGD and mini-batch* (p8): batch vs stochastic vs mini-batch, noisy paths, schedule ★, "new data, no restart". (2)
9. *Summary*: one-page cheat sheet. (1)
Statement slides at three turning points: "step = slope × learning rate"; "the learning rate decides stable or not"; "first order = GD, second order = Newton".

**Part B: problems** (each problem frame followed at once by its solution frames)

## 6. The ten problems (numbers fixed by `verify_problems.py` before writing)

| # | Problem | Level | Tests |
|---|---|---|---|
| M1 | Stationary points of J(x) = x⁴ − 6x³ + 8x² (the curve from the notes): find them, classify with J'', then run GD from x₀ = 1 and x₀ = 1.5 and see which minimum each reaches | ★★ | stationary points, local vs global |
| M2 | GD by hand on f(x) = (x − 3)² + 2 with α = 0.1, 0.5, 0.9, 1.1: derive e_{k+1} = (1 − 2α)e_k, give the convergence condition, and say which runs converge, oscillate or diverge | ★★ | step size, stability |
| M3 | J(x₁, x₂) = x₁² + 4x₂²: gradient, two GD steps from (2, 1), the largest stable α, and why the path zig-zags compared with x₁² + x₂² | ★★ | 2-D gradient, conditioning |
| M4 | Scaling and shifting the cost (note p18): show that cJ + d has the same minimiser as J, that one GD step on cJ equals a step on J with learning rate cα, and what this means for SSR vs MSE = SSR/(2m) | ★★ | derivation, reasoning |
| M5 | GD vs Newton on f(x) = (x − 3)² (Newton exact in one step; GD needs k steps for 10⁻⁶) and on x⁴ − 6x³ + 8x² from x₀ = 1.4, where f'' < 0 so Newton runs to the maximum while GD goes downhill | ★★★ | order of convergence, comparing methods |
| A1 | **Biology / health:** the class's weight-height data (0.5, 1.4), (2.3, 1.9), (2.9, 3.2): GD on the intercept with the slope fixed at 0.64, three steps by hand, then both parameters, checked against the least-squares closed form (p16) | ★★ | the class example, done properly |
| A2 | **Real estate:** multiple linear regression on three houses (size, bedrooms → price) with MSE: one GD iteration from w = 0, then why unscaled size forces a tiny α, and the fix by scaling | ★★ | multivariable GD, feature scaling |
| A3 | **Streaming sensor data:** fit y = wx on arriving readings with SGD (one sample at a time, one epoch by hand) vs batch GD vs mini-batch of 2; what "new data, no restart" means | ★★ | SGD, mini-batch |
| A4 | **Engineering design:** a cylindrical can with volume 500 cm³; minimise the surface area A(r) = 2πr² + 1000/r with GD; compare with the exact r* = (250/π)^{1/3} | ★★ | 1-D GD in a design problem |
| A5 | **Classification (logistic regression, the note's "squiggle"):** pass/fail against study hours with a sigmoid and log-loss; why there is no closed form; two GD steps by hand | ★★★ | GD where it is the only option |

## 7. Figures

Python with the `paper-navy-beamer` figure style (`figstyle.py`):
- SSR vs intercept with GD steps shrinking near the bottom (the p6 picture, recomputed)
- the fitted line moving over the data as GD runs (a step-by-step overlay animation, the only `\pause`-style overlay in Part A)
- learning-rate regimes on (x − 3)²: four small panels
- J = x₁² + x₂² vs x₁² + 4x₂² contours with GD paths; gradient arrows ⟂ contours
- x⁴ − 6x³ + 8x² with GD from two starts
- GD vs Newton convergence on a semilog plot
- batch vs SGD vs mini-batch paths and loss curves
- the can-design curve A(r); the logistic sigmoid fit

TikZ: roadmap, steepest-descent geometry (unit circle of directions), the algorithm flow (steps i–vi from the notes), the Taylor "first order vs second order" picture.

## 8. Project layout

```
assignment-revisit/
├── PLAN.md                       (this file, not submitted)
├── build.ps1                     (build + zip + clean-room test, not submitted)
├── 2105032.pdf                   (deliverable)
├── 2105032_LaTeX_Source.zip      (deliverable)
└── 2105032_LaTeX_Source/
    ├── 2105032.tex   preamble.tex (paper-navy theme)   README.md
    ├── sections/ (part A files, problems/M1..A5)
    ├── figures/generated/ (PDFs)   figures/tikz/
    ├── scripts/ make_figures.py  verify_problems.py  figstyle.py  requirements.txt
    └── bibliography/references.bib  (small: Chapra, Nocedal & Wright, Bottou et al.; all verified before)
```

## 9. Who does what

- **Codex subagent:** `verify_problems.py`. It computes every number in §6 with assertions, so the slides are written from checked values.
- **Sonnet subagents:** `make_figures.py` (§7, using `figstyle.py`), and a final overflow/visual review of the contact sheets.
- **Main thread:** the plan, all slide text and maths, problem solutions from the verified numbers, the theme set-up, the build and packaging.

## 10. Order of work

1. Copy the theme from the skill; set up the folder.
2. Write `verify_problems.py` (Codex) and `make_figures.py` (Sonnet) in parallel.
3. Write Part B from the verified numbers first (most of the marks), then Part A.
4. Build, fix overflow, check every page visually, then package and run the clean-room test.

## 11. Decision for you

Both decks will exist side by side. Moodle takes one PDF and one zip, so before the deadline you choose
which pair to upload: `Assignment/` (first deck) or `assignment-revisit/` (this one). The file names are the same.
