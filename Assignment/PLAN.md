# Implementation Plan — CSE 402 Assignment: Gradient Methods (Beamer)

**Student roll:** 2105032  **Deadline:** Friday 2 Oct 2026, 11:59 PM  **Status:** PLAN ONLY, waiting for approval

All problem numbers below were checked numerically (NumPy) while writing this plan. The final build will re-check them with `scripts/verify_problems.py`.

---

## A. Verified topic calculation

| Step | Value |
|---|---|
| Roll number | 2105032 |
| Last 3 digits | 032 = **32** |
| 32 mod 15 | **2** (15 × 2 = 30, remainder 2) |
| Topic ID = 2 + 1 | **3** |
| Topic 3 in the table | **Gradient Methods** ✅ |

This matches the expected topic. The deck will show the calculation on its second slide, so the grader sees straight away that the topic is correct.

**Topic-boundary rule (the zero-marks risk):** the neighbouring topics are *Golden-Section Search* (1), *Newton's Method* (2) and *Constrained Optimization* (4). They belong to other students. The deck may **mention** them only as contrast or as tools (for example, "a 1-D line search could use golden-section"), and never **teach** them. Every section must stay about first-order (gradient-based) methods.

---

## B. Presentation narrative (the teaching story)

The deck follows **one central idea**:

> *Gradient descent is forward Euler applied to the gradient-flow ODE ẋ = −∇f(x).*
> That single idea explains the step-size limit (Euler's stability interval), the slow convergence (stiffness = ill-conditioning), and why momentum and preconditioning help.

This connects to material the class already knows (Euler's method, stability, order of convergence). It is also where most of the "wow factor" comes from, because ordinary lecture notes rarely make this link.

The story runs in this order:

1. **Problem.** We want min f(x). Solving ∇f = 0 directly is a nonlinear system and gets hard when n is large.
2. **Intuition.** The gradient points uphill fastest (proved with Cauchy–Schwarz). Twist: "fastest" depends on the norm you choose, and this idea leads to preconditioning later.
3. **Derivation.** GD in three views: (i) Taylor with a proximity penalty; (ii) minimising a quadratic upper model; (iii) forward Euler on gradient flow.
4. **Geometry.** Contours, gradient ⟂ level sets, trajectories.
5. **Step size.** Exact analysis on quadratics: e_{k+1} = (I − αA)e_k, which gives 0 < α < 2/λ_max. This is the Euler stability interval.
6. **Function classes.** L-smooth, convex, μ-strongly convex. Sandwich picture; κ = L/μ.
7. **Convergence.** Descent lemma, then rates for nonconvex, convex and strongly convex f, plus PL. Order-of-convergence view.
8. **Why it is slow.** Conditioning, the zig-zag and orthogonal steps, the Kantorovich bound, and scaling/preconditioning as the fix.
9. **Failure modes.** Divergence, saddles, plateaus, non-Lipschitz gradients, nonsmoothness, finite precision, finite-difference gradients.
10. **Adaptive steps.** Exact line search, Armijo backtracking, Barzilai–Borwein.
11. **Acceleration.** Heavy-ball momentum, Nesterov, and the lower bound showing Nesterov is optimal.
12. **Stochastic GD.** Noise ball, Robbins–Monro conditions, mini-batching.
13. **Modern view.** Automatic differentiation's cheap-gradient principle, Adam as a diagonal preconditioner, edge of stability. Kept short and factual.
14. **Applications, then the 10 problems with solutions.**

---

## C. Section / slide architecture

Target size is about **85–95 frames**: about 55 on theory, 30–34 on problems, and 5 of front and back matter. Overlays (`\pause`, `\only`) add pages but not frames.

### 0. Front matter (3 frames)
- Title slide (name, roll, course, topic).
- **Topic verification:** 032 mod 15 + 1 = 3.
- Roadmap (TikZ flow diagram of the story above).

### 1. Motivation & the optimization problem (3)
- 1.1 Optimization everywhere: ML training, structural energy, economics, data fitting. One slide.
- 1.2 Formal problem: min_{x∈ℝⁿ} f(x); local vs global minima; ∇f = 0 as a necessary condition.
- 1.3 Why iterate rather than solve ∇f = 0? When n is huge, first-order methods cost O(n) memory per iteration. Short historical note: Cauchy, 1847.

### 2. Gradient & directional derivative (4)
- 2.1 Definitions: gradient and directional derivative D_d f = ∇fᵀd.
- 2.2 **Theorem:** −∇f/‖∇f‖ is the steepest-descent unit direction. **Proof** via Cauchy–Schwarz.
- 2.3 ∇f ⟂ level sets (proof sketch). Figure: contour plot with a quiver of gradient vectors.
- 2.4 **Wow:** steepest descent depends on the norm. In the ‖·‖_P norm the direction becomes −P⁻¹∇f, so preconditioning is still steepest descent, just in a different geometry. Only a remark is made about Newton; it is not taught.

### 3. Deriving gradient descent (4)
- 3.1 First-order Taylor model plus the penalty (1/2α)‖x − x_k‖²; minimising it gives x_{k+1} = x_k − α∇f(x_k).
- 3.2 **Gradient flow** ẋ = −∇f(x). GD is forward Euler with h = α. Along the flow, d/dt f(x(t)) = −‖∇f‖² ≤ 0.
- 3.3 Algorithm box: pseudocode and stopping criteria. Uses a relative gradient norm plus a cap on iterations, and explains why a test like |f_{k+1} − f_k| < tol alone is dangerous.
- 3.4 Vectorized NumPy implementation (about 10 lines, using `listings`). Cost per iteration is one gradient evaluation plus O(n).

### 4. Geometric interpretation (2)
- 4.1 Trajectory on a contour map, **stepped with overlays** (one iterate per click).
- 4.2 Each step leaves perpendicular to the current contour. What the path looks like on a round bowl versus an elongated one (sets up §8).

### 5. Step-size analysis & stability (5)
- 5.1 Model problem f = ½xᵀAx − bᵀx with A symmetric positive definite: e_{k+1} = (I − αA)e_k. Eigen-decomposition gives one independent 1-D recursion per eigen-direction.
- 5.2 **Theorem:** convergence for every x₀ ⇔ 0 < α < 2/λ_max. Rate per direction |1 − αλ_i|.
- 5.3 Figure: |1 − αλ| against α for λ_min and λ_max, with α* = 2/(μ+L) and ρ* = (κ−1)/(κ+1) marked.
- 5.4 Figure: four trajectories (too small, optimal, near-critical oscillation, divergent).
- 5.5 **Wow:** this is exactly the stability interval of explicit Euler, |1 − hλ| < 1. An ill-conditioned objective is a *stiff* gradient flow. TikZ: Euler stability disk in the complex plane with the scaled eigenvalues −αλ_i plotted.

### 6. Function classes: smoothness, convexity, strong convexity (3)
- 6.1 L-smooth (Lipschitz gradient); convexity; μ-strong convexity. Equivalent forms using the Hessian's eigenvalues.
- 6.2 TikZ "sandwich": f squeezed between a quadratic lower bound (curvature μ) and a quadratic upper bound (curvature L) at x.
- 6.3 Condition number κ = L/μ, what it means, and examples.

### 7. Convergence analysis (6)
- 7.1 **Descent lemma**, with proof: f(y) ≤ f(x) + ∇fᵀ(y − x) + (L/2)‖y − x‖².
- 7.2 Sufficient decrease: f(x_{k+1}) ≤ f(x_k) − α(1 − αL/2)‖∇f_k‖². This gives α ∈ (0, 2/L), and α = 1/L is the best choice. (Problem M2 asks for this proof.)
- 7.3 **Nonconvex:** min_{k<K} ‖∇f_k‖² ≤ 2L(f₀ − f*)/K. Only stationarity is guaranteed.
- 7.4 **Convex:** f(x_k) − f* ≤ L‖x₀ − x*‖²/(2k), i.e. O(1/k). Proof sketch.
- 7.5 **Strongly convex:** ‖x_k − x*‖² ≤ (1 − μ/L)^k ‖x₀ − x*‖², i.e. linear (geometric) convergence. **PL inequality**: the same linear rate holds without convexity (Karimi–Nutini–Schmidt).
- 7.6 Table of complexity and order: iterations to reach ε are O(κ log 1/ε) (strongly convex) and O(L/ε) (convex). GD is only *linearly* convergent, with order p = 1. **Empirical order check**: plot log e_{k+1} against log e_k, which gives slope 1. This ties to the course's "order of convergence" material.

### 8. Why convergence can be slow: conditioning & zig-zag (4)
- 8.1 Exact line search on a quadratic: α_k = g_kᵀg_k / g_kᵀAg_k. **Proof** that consecutive gradients are orthogonal, which causes the zig-zag.
- 8.2 Kantorovich bound: f_{k+1} − f* ≤ ((κ−1)/(κ+1))² (f_k − f*). The closed-form worst case on ½(x² + γy²) attains it (Problem M4).
- 8.3 Figure: zig-zag for κ = 10 and κ = 100 side by side, plus semilog error curves against the theoretical bound.
- 8.4 Fix: **scaling / preconditioning.** Contours before and after feature centering (links to A1), and why ML practitioners normalise data.

### 9. Failure modes & pathological cases (5)
- 9.1 Divergence and oscillation: α > 2/L. **Non-globally-Lipschitz gradient:** for f = x⁴, local L = 12x². No fixed α works from every start, and GD diverges from far away.
- 9.2 **Saddle points:** f = x⁴ + y⁴ − 4xy (the same function as M1). A start exactly on the stable manifold converges to the saddle. Random starts escape almost surely (Lee et al., 2016). 3-D surface figure.
- 9.3 **Plateaus / vanishing gradients:** the flat region of the nonlinear least-squares loss in A4. The Rosenbrock banana valley.
- 9.4 **Nonsmooth f:** f = |x| with a constant step never converges; it cycles in a band of width about α. This is why subgradient methods need diminishing steps.
- 9.5 **Finite precision:** the ‖∇f‖ tolerance cannot go below roughly the noise floor. **Finite-difference gradients:** error ≈ Lh/2 + 2ε_mach|f|/h, so the best h is about √ε_mach. This is a truncation-versus-roundoff trade-off, a core numerical-methods lesson.

### 10. Line search & adaptive step sizes (3)
- 10.1 Exact line search: what it costs and when it is worth using. A 1-D search such as golden-section *could* be used; only mentioned.
- 10.2 **Armijo backtracking:** the condition, the algorithm, and why it ends with α ≥ β/L (proof from the descent lemma). It removes the need to know L.
- 10.3 **Barzilai–Borwein** step: α_k = s_kᵀs_k / s_kᵀy_k. A secant-style estimate of curvature. Nonmonotone, often much faster in practice. Figure comparing GD, Armijo and BB.

### 11. Momentum & Nesterov acceleration (5)
- 11.1 Heavy ball (Polyak): x_{k+1} = x_k − α∇f + β(x_k − x_{k−1}). Physical picture: a ball with inertia and friction, i.e. a discretised 2nd-order ODE.
- 11.2 Analysis on quadratics: the characteristic polynomial r² − (1 + β − αλ)r + β = 0. The optimal parameters give rate (√κ−1)/(√κ+1). (Problem M5.)
- 11.3 Nesterov accelerated gradient: the look-ahead gradient. For convex f, f(x_k) − f* ≤ 2L‖x₀ − x*‖²/(k+1)². Stated with citation; a full proof is out of scope.
- 11.4 **Lower bound:** no first-order method can beat Ω(1/k²) (convex) or Ω((√κ−1)/(√κ+1))^k in the worst case (Nesterov). So Nesterov is *optimal*. Explains the non-monotone "ripples".
- 11.5 Figure: GD vs heavy ball vs Nesterov on a quadratic with κ = 100 and on Rosenbrock, with trajectories and semilog error curves.

### 12. Stochastic gradient descent (3)
- 12.1 Finite-sum objective (1/N)Σf_i. An unbiased stochastic gradient costs O(1) instead of O(N).
- 12.2 Constant step converges to a **noise ball** of radius O(α). Decaying steps under the Robbins–Monro conditions Σα_k = ∞, Σα_k² < ∞. Mini-batch variance is σ²/B. (Problem A5 does this exactly in 1-D.)
- 12.3 Figure: trajectories and error with a constant step versus α_k ∝ 1/k.

### 13. Modern computational perspective (2)
- 13.1 **Cheap gradient principle:** reverse-mode automatic differentiation computes ∇f for a small constant multiple of the cost of f (Griewank & Walther). This is why first-order methods dominate large-scale computing. Also: vectorisation and running on GPUs.
- 13.2 Adam and RMSProp read as *diagonal adaptive preconditioning* (a link back to §2.4). The **edge of stability**: in neural-network training, sharpness is observed to sit near 2/α (Cohen et al.). Presented as an empirical observation, not a theorem.

### 14. Applications overview & summary (2)
- 14.1 One-slide map of the 5 application problems.
- 14.2 Cheat-sheet table: each method with its step rule, rate, cost per iteration and main failure mode.

### Part B. Problem set (10 problems, about 30–34 frames)
Each problem has **1 problem frame followed immediately by 2–3 solution frames**, titled "Problem Mk — Solution (i/n)". Problems M1–M5 come first, then A1–A5.

### Back matter (2)
- References (BibTeX, `\tiny`, `allowframebreaks`).
- Closing / thank-you slide.

---

## D. The 10 problems

### Pure mathematical problems (5)

**M1. Stationary points & landscape of f(x,y) = x⁴ + y⁴ − 4xy.** *Difficulty: ★★☆*
- (a) Find ∇f and all stationary points. (b) Classify them with the Hessian. (c) Do one GD step from (1, 0) with α = 0.1 and check that f decreases. (d) What is the largest α for which GD is locally stable near the minimisers?
- Answers: stationary points are (0,0), (1,1) and (−1,−1). The Hessian eigenvalues are {−4, 4} at the origin (a saddle) and {8, 16} at (±1,±1) (minima, f = −2). The step gives x₁ = (0.6, 0.4), and f goes from 1 to −0.8048. The local limit is α < 2/16 = 0.125. Remark: GD started exactly at (0,0) stays at the saddle, which leads into §9.2.
- *Purpose:* gradients, stationary points, classification, local stability.

**M2. Proof: the descent lemma and the O(1/√K) nonconvex rate.** *Difficulty: ★★★*
- (a) Prove the descent lemma for L-smooth f, using the integral form of Taylor's theorem and Cauchy–Schwarz. (b) Show that f(x_{k+1}) ≤ f(x_k) − α(1 − αL/2)‖∇f_k‖². (c) Find the α that maximises the guaranteed decrease (α = 1/L). (d) Prove min_{k<K} ‖∇f_k‖ ≤ √(2L(f₀ − f*)/K).
- *Purpose:* rigorous derivation, why the step-size window is (0, 2/L), and the rate without convexity.

**M3. Step-size stability on a coupled quadratic.** *Difficulty: ★★☆*
- f(x) = ½xᵀAx − bᵀx with A = [[3,1],[1,3]] and b = (4,4). (a) Find x* (answer: (1,1)). (b) Derive the error recursion and the eigen-decomposition (λ = 2 and 4, eigenvectors (1,−1) and (1,1)). (c) Find the stability range (0 < α < 0.5). (d) Find the best α* = 1/3 and the rate ρ = 1/3. (e) How many iterations reduce the error by 10⁶ (answer: 13)? (f) Where does α = 0.6 diverge, and how? Answer: along (1,1), with factor −1.4, so the error grows while flipping sign each step.
- *Purpose:* exact stability analysis and comparing step choices.

**M4. Exact line search, orthogonal zig-zag, and conditioning.** *Difficulty: ★★★*
- f = ½(x² + γy²), γ ≥ 1, x₀ = (γ, 1). (a) Derive the exact step for quadratics. (b) Prove that consecutive gradients are orthogonal. (c) Prove by induction that x_k = γr^k and y_k = (−r)^k with r = (γ−1)/(γ+1). (d) Show f_k = r^{2k} f₀, so the Kantorovich bound is attained. (e) Iterations to reduce f by 10⁶: 35 for γ = 10, 346 for γ = 100.
- The closed form is a standard textbook example (Boyd & Vandenberghe, §9.3); the solution will credit it.
- *Purpose:* conditioning, convergence rate, a proof by induction.

**M5. Heavy-ball momentum vs plain GD: a spectral comparison.** *Difficulty: ★★★*
- 1-D model f = ½λx² with λ ∈ [μ, L]. (a) Derive the heavy-ball characteristic polynomial. (b) Show that when the roots are complex, |r| = √β for every λ. (c) With μ = 1 and L = 100: α = 4/121 ≈ 0.0331, β = 81/121 ≈ 0.669. Verify that every λ ∈ [1, 100] lies in the complex (or double-root) regime, since (1−√β)² = 4/121 and (1+√β)² = 400/121. (d) Compare iterations to reach 10⁻⁶: GD about 691, heavy ball about 69, a √κ = 10× speed-up.
- **Subtlety for the wow factor:** at λ = μ and λ = L the roots are *repeated*, so the error behaves like k·r^k, and the rate is only asymptotic.
- *Purpose:* comparing methods, the eigenvalue view of an iteration, acceleration.

### Real-life application problems (5)

**A1. Machine learning: linear regression by GD, and why feature scaling matters.** *Difficulty: ★★☆*
- Data (1,2), (2,3), (3,5); model ŷ = wx + b; MSE loss with ½.
- (a) Gradient; first step from (0,0) with α = 0.1. The gradient is (−23/3, −10/3), so (w,b) → (0.767, 0.333). (b) Hessian [[14/3, 2],[2, 1]], eigenvalues ≈ 5.546 and 0.120, **κ ≈ 46.1**, and α_max ≈ 0.361. (c) After centering x, the Hessian is diag(2/3, 1) and κ = 1.5. (d) The optimum is w = 1.5, b = 1/3.
- *Purpose:* shows data preprocessing is preconditioning.

**A2. Structural mechanics: equilibrium of a spring chain by energy minimisation.** *Difficulty: ★★☆*
- Wall → spring k₁ → mass 1 → spring k₂ → mass 2, with load F on mass 2. Potential energy Π(u) = ½uᵀKu − fᵀu.
- (a) Show ∇Π = 0 is force balance. (b) With k₁ = 3, k₂ = 1 (kN/m) and F = 1 kN: u* = (1/3, 4/3). λ(K) ≈ 0.697 and 4.303, κ ≈ 6.17, α_max ≈ 0.465, α* = 0.4, rate ≈ 0.721. (c) A stiff support, k₁ = 1000, gives κ ≈ 1002, α_max ≈ 0.002 and a rate of about 0.998. Link this to *stiff systems* and to preconditioning.
- *Purpose:* physical meaning of the Hessian, stiffness = ill-conditioning (an explicit request in the brief).

**A3. Economics: two-product profit maximisation by gradient ascent.** *Difficulty: ★★☆*
- P(x,y) = 100x + 80y − 2x² − 2y² − 2xy (substitute goods).
- (a) Analytic optimum (20, 10), P = 1400. (b) Hessian of −P has eigenvalues 2 and 6. Stable if α < 1/3; α* = 0.25 with rate 0.5. (c) Three hand iterations from (0,0): (25,20) → (15,7.5) → (21.25,12.5). Check that the error shrinks by exactly 0.5 per step (0.5³ over three steps), because both eigen-factors have magnitude 0.5. (d) Interpretation: step size as a "managerial adjustment speed"; too aggressive a step oscillates.
- *Purpose:* ascent vs descent, a hand calculation checked against the theory.

**A4. Parameter estimation: Newton's law of cooling (nonlinear least squares).** *Difficulty: ★★★*
- Normalised temperature data y(1) = 0.6 and y(2) = 0.36; model e^{−kt}; J(k) = ½Σ(e^{−kt_i} − y_i)².
- (a) J′(k). (b) Two GD steps from k = 0 with α = 0.5: J′(0) = −1.68, k₁ = 0.84 (overshoot), k₂ ≈ 0.771. It converges to k* = ln(5/3) ≈ 0.5108. (c) Curvature varies a lot: J″(0) = 7.96 but J″(k*) = 0.878. (d) **Plateau:** for large k, J′ → 0 (for example J′(10) ≈ 2.7×10⁻⁵) and J is **nonconvex** there, so GD started far away stalls.
- *Purpose:* a real nonconvex failure mode (plateau / vanishing gradient) and local vs global step-size limits.

**A5. Signal processing: streaming sensor calibration with SGD.** *Difficulty: ★★☆*
- Estimate a true value θ from noisy readings ξ_k = θ + noise (variance σ²) by minimising ½E[(x − ξ)²].
- (a) SGD update: x_{k+1} = (1 − α_k)x_k + α_kξ_k. (b) Prove that α_k = 1/(k+1) reproduces the **running sample mean** exactly, with variance σ²/k → 0. (c) For constant α, derive the stationary variance ασ²/(2 − α): a noise ball that never shrinks. (d) Numerical run on 5 given readings, comparing the two schedules. (e) Link to the Robbins–Monro conditions.
- *Purpose:* SGD theory in a setting small enough to solve exactly by hand.

**Coverage check:** gradient calculation (M1, A1, A4); stationary points (M1, A3); derivation/proof (M2, M4, A5); step-size and stability (M1, M3, A2, A3); convergence and conditioning (M3, M4, A1, A2); comparing choices (M3, M5, A5).

---

## E. Required visualizations

### Python / Matplotlib (generated by `scripts/make_figures.py`; output PDF for vectors and PNG for dense plots)

| # | Figure | Used in |
|---|---|---|
| P1 | Contours plus a quiver of −∇f on x⁴ + y⁴ − 4xy (saddle and two minima) | §2.3, M1 |
| P2 | 3-D surface of the same function, showing the saddle | §9.2 |
| P3 | GD trajectory **frames** (one PNG per iterate) for the overlay animation | §4.1 |
| P4 | Four step-size trajectories (small / optimal / near-critical / divergent) | §5.4, M3 |
| P5 | \|1 − αλ\| against α for λ_min and λ_max; α* and ρ* marked | §5.3 |
| P6 | Zig-zag with exact line search, κ = 10 vs 100 | §8.3, M4 |
| P7 | Semilog error curves compared with the theoretical bounds | §7.6, §8.3 |
| P8 | Empirical order of convergence: log e_{k+1} against log e_k | §7.6 |
| P9 | Contours before and after feature centering | §8.4, A1 |
| P10 | GD vs heavy ball vs Nesterov: trajectories and error curves (quadratic + Rosenbrock) | §11.5, M5 |
| P11 | GD vs Armijo vs Barzilai–Borwein error curves | §10.3 |
| P12 | SGD: constant vs decaying step (noise ball) | §12.3, A5 |
| P13 | A4 loss curve J(k), showing the plateau and nonconvex region | §9.3, A4 |
| P14 | Finite-difference gradient error against h (V-shaped log-log curve) | §9.5 |

All scripts use a fixed random seed, the same palette in every figure, and colourblind-safe colours. Each figure is sized for a 16:9 frame.

### TikZ (drawn natively in the deck, so it stays sharp and editable)
- T1 Roadmap / story flow diagram.
- T2 Steepest-descent geometry: unit circle of directions, ∇f, and the angle θ in the Cauchy–Schwarz argument.
- T3 1-D descent lemma: f, its tangent, and the quadratic upper bound touching at x_k, with the GD step as the minimiser of that bound.
- T4 Sandwich picture of μ-strong convexity and L-smoothness.
- T5 Explicit-Euler stability disk in the complex plane with the −αλ_i markers.
- T6 Spring-chain diagram (A2).
- T7 Heavy-ball "ball with friction" sketch (optional).

### Beamer overlays (reliable in every PDF viewer)
- §4.1 GD iterates appear one per click (`\only<n>{\includegraphics{frame_n}}`).
- Proofs reveal step by step with `\pause` / `\uncover`.
- In each solution, the final answer is revealed last.

**Deliberately avoided: the `animate` package.** It only plays in Adobe Acrobat and fails silently in browsers, SumatraPDF and most graders' viewers. Overlays give the same step-by-step effect everywhere.

---

## F. Recommended LaTeX / project architecture

```
Assignment/
├── PLAN.md                      (this file, not submitted)
├── build.ps1                    (build + zip helper, not submitted)
└── 2105032_LaTeX_Source/        (becomes 2105032_LaTeX_Source.zip)
    ├── 2105032.tex              primary document (compiles directly to 2105032.pdf)
    ├── preamble.tex             packages, theme, colours, macros, theorem/box envs
    ├── sections/
    │   ├── 01_motivation.tex    … 14_summary.tex   (one file per section in §C)
    │   └── problems/
    │       ├── M1.tex … M5.tex
    │       └── A1.tex … A5.tex  (problem frame + solution frames together)
    ├── figures/
    │   ├── generated/           Matplotlib outputs (.pdf/.png), committed
    │   └── tikz/                standalone .tikz snippets pulled in with \input
    ├── scripts/
    │   ├── make_figures.py      regenerates every figure in figures/generated
    │   ├── verify_problems.py   recomputes every number used in the solutions
    │   └── requirements.txt     numpy, matplotlib
    ├── bibliography/
    │   └── references.bib
    └── README.md                build instructions + file map
```

Design decisions:
- **Primary file named `2105032.tex`, not `main.tex`.** Running `pdflatex 2105032.tex` then produces `2105032.pdf` with no renaming step, so the grader's compile gives exactly the submitted filename. (If you prefer `main.tex`, the README will state the rename.)
- **Engine: pdfLaTeX.** It is the most portable across MiKTeX and TeX Live. No `fontspec`, no shell-escape, no `minted`, which needs Python and Pygments at compile time. Code is typeset with `listings`.
- **Bibliography: BibTeX, not biber.** A mismatch between the biblatex and biber versions is a common failure on graders' machines. The zip will also include the generated `.bbl` as a fallback.
- **Theme:** a built-in Beamer theme with a custom colour palette, using only packages that ship with standard distributions: `beamer`, `tikz`, `pgfplots` (optional), `tcolorbox`, `amsmath/amssymb/amsthm`, `listings`, `booktabs`. Metropolis is installed locally and looks modern, but it depends on the font package under pdfLaTeX; I will use it only if a clean test build passes, otherwise the default theme. (Your personal navy/gold "diffusion" Beamer template is also available if you want it; just say so.)
- **Figures are pre-generated and committed**, so the deck compiles without Python. The scripts are included to show where every figure comes from.
- The zip excludes `.aux/.log/.nav/.out/.snm/.toc` and includes the `.bbl`.

Environment check (done): MiKTeX 24.4 has pdflatex, latexmk, bibtex and biber, plus beamer, tikz, pgfplots, tcolorbox and metropolis. Python 3.12 has NumPy 2.1.3 and Matplotlib 3.9.2. The toolchain is ready.

---

## G. References to consult / cite

Only well-known, real sources. **Bibliographic details (edition, pages, year) will be checked before they go into `references.bib`; nothing is cited for a claim it doesn't support.**

Textbooks
1. S. C. Chapra & R. P. Canale, *Numerical Methods for Engineers*, McGraw-Hill. The course's likely text; its topic list matches the chapter on multidimensional unconstrained optimization. Used for the baseline and the class-style notation.
2. J. Nocedal & S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006. Line search, Armijo/Wolfe, the steepest-descent rate (Kantorovich).
3. S. Boyd & L. Vandenberghe, *Convex Optimization*, Cambridge Univ. Press, 2004. Descent-method analysis, the closed-form zig-zag example (M4).
4. Y. Nesterov, *Lectures on Convex Optimization*, 2nd ed., Springer, 2018. Lower complexity bounds, accelerated gradient.
5. S. Bubeck, "Convex Optimization: Algorithms and Complexity," *Foundations and Trends in ML*, 2015. Clean proofs of the rates.
6. A. Griewank & A. Walther, *Evaluating Derivatives*, 2nd ed., SIAM, 2008. The cheap gradient principle.

Original and research papers
7. A. Cauchy, 1847, *Comptes Rendus*: the origin of steepest descent.
8. L. Armijo, 1966, *Pacific J. Math.*: backtracking condition.
9. B. T. Polyak, 1964, *USSR Comput. Math. & Math. Phys.*: heavy-ball method.
10. Y. Nesterov, 1983, *Soviet Math. Doklady*: O(1/k²) method.
11. H. Robbins & S. Monro, 1951, *Ann. Math. Statist.*: stochastic approximation.
12. J. Barzilai & J. M. Borwein, 1988, *IMA J. Numer. Anal.*: two-point step size.
13. H. Karimi, J. Nutini & M. Schmidt, 2016, ECML-PKDD: PL-condition linear convergence.
14. J. D. Lee, M. Simchowitz, M. I. Jordan & B. Recht, 2016, COLT: GD converges to minimisers (avoids saddles almost surely).
15. W. Su, S. Boyd & E. Candès, 2016, *JMLR*: ODE model of Nesterov acceleration.
16. L. Bottou, F. E. Curtis & J. Nocedal, 2018, *SIAM Review*: optimisation for large-scale ML (SGD).
17. D. P. Kingma & J. Ba, 2015, ICLR: Adam.
18. J. Cohen et al., 2021, ICLR: edge of stability.

Intuition resource (for designing figures, optional to cite)
19. G. Goh, "Why Momentum Really Works," *Distill*, 2017.

---

## H. Requirement → deliverable checklist

| Requirement (instructions.txt) | Where it is met |
|---|---|
| Correct topic via the formula | §A; deck frame 0.2; every section is about gradient methods |
| Stay inside the topic | Topic-boundary rule in §A; Newton, golden-section and constrained methods only mentioned |
| LaTeX only (Beamer) | Whole project is Beamer; no Word or Docs files |
| "Wow factor": deeper proofs / error bounds | §2.2, §7.1–7.5, §8.1–8.2, §10.2; M2, M4, M5 |
| "Wow factor": instability, pathology, stiff behaviour | §5.5 (Euler/stiffness), §9 (5 frames), A2 (stiff spring), A4 (plateau) |
| "Wow factor": modern adaptations (vectorised, adaptive step, high-D) | §3.4 NumPy, §10 Armijo/BB, §12 SGD, §13 autodiff/Adam |
| "Wow factor": unique visuals, step-by-step logic | §E: P1–P14, T1–T7, overlay animation §4.1 |
| Order of convergence | §7.6 (linear, p = 1, empirical order plot P8) |
| Stability regions | §5.2–5.5 (α window, Euler stability disk T5) |
| Computational complexity | §3.4, §7.6, §12.1, §13.1 |
| Failure modes | §9 |
| 5 pure mathematical problems | M1–M5 |
| 5 real-life application problems | A1–A5 (ML, structural mechanics, economics, parameter estimation, signal processing) |
| Full step-by-step solution right after each problem | Problem frame followed at once by 2–3 solution frames |
| PDF named `2105032.pdf` | Primary file `2105032.tex` produces it directly |
| Zip named `2105032_LaTeX_Source.zip` | `build.ps1` creates it from the folder |
| Zip contains primary .tex, all images/diagrams, style/bib files | sections/, figures/generated, figures/tikz, preamble.tex, references.bib (+ .bbl) |
| Source must compile to the submitted PDF | pdfLaTeX, no shell-escape, pre-generated figures; clean-room test build from the unzipped archive before submitting |
| Two separate Moodle links | Reminder in README: PDF → "PDF File Submission", zip → "LaTeX File Submission" |

---

## I. Recommended implementation order

1. **Skeleton + style:** `2105032.tex`, `preamble.tex`, empty section stubs. Compile once to prove the toolchain and theme work.
2. **`verify_problems.py`:** lock in every number for M1–M5 and A1–A5 before any slide text is written.
3. **`make_figures.py`:** generate P1–P14 and check them visually.
4. **Problems (Part B) first.** They are worth 15 of the 20 marks, so write M1–M5 then A1–A5, with solutions taken from the verified numbers.
5. **Theory sections §1–§8**, the core narrative, including proofs.
6. **§9–§13** failure modes, adaptive steps, acceleration, SGD, modern view.
7. **TikZ diagrams** T1–T7.
8. **References + summary + title/verification frames.**
9. **Quality pass:** re-derive every proof line and check the overlays. Look for overfull boxes and text that is too dense (at most about 8 lines per frame).
10. **Packaging:** `build.ps1` does a clean build, then zips `2105032_LaTeX_Source/` → `2105032_LaTeX_Source.zip`. Then **unzip into a temp folder and recompile from scratch** to confirm it compiles, and copy out `2105032.pdf`.

Parallelism: steps 2–3 (the scripts) and the TikZ drawings can be done by Sonnet subagents while the main thread writes the slides. Proofs and final solutions stay in the main thread for correctness.

---

## Risks & ambiguities

1. **Deadline pressure.** The deadline is tomorrow night. About 90 frames is achievable, but if time runs short, cut in this order: T7, P11, §13.2, P14. Never cut problems or core proofs.
2. **Topic leakage (zero-marks risk).** Newton's method and constrained optimization are other students' topics. Keep them to one-line contrasts.
3. **"Gradient Methods" scope.** Could the grader expect Chapra's "steepest ascent" framing (maximisation, with an optimal step found by 1-D search)? The deck covers it explicitly: ascent is shown alongside descent (§3, A3), and exact line search is in §8.1.
4. **"Formatted clearly after each problem slide".** Read as: solution frames come *immediately after* each problem, not in an appendix. This plan follows that reading.
5. **"All raw image files … (.tikz)".** TikZ is kept in separate `.tikz` files in `figures/tikz/` (not only inline), so this literal requirement is met.
6. **Compile portability.** The grader may use TeX Live or Overleaf rather than MiKTeX. Standard packages only, no shell-escape, and the `.bbl` is shipped, which keeps this safe. A test compile on Overleaf is recommended if possible.
7. **Heavy-ball rate subtlety (M5).** The rate (√κ−1)/(√κ+1) is asymptotic because of the repeated roots at the extreme eigenvalues. The solution states this precisely rather than over-claiming.
8. **Heavy ball on non-quadratics.** Its global acceleration is not guaranteed for general strongly convex f (known counterexamples exist). The deck will claim acceleration only for quadratics, or will attribute it to Nesterov's method for the general case.
