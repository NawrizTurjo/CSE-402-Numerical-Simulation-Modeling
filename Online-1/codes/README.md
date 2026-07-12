# CSE-402 Online Lab 1 — Practice Codebase
## Quick Navigation

---

### 📁 File Index

| File | Type | Purpose |
|------|------|---------|
| [`cheatsheets/00_exam_cheatsheet.py`](cheatsheets/00_exam_cheatsheet.py) | 🐍 Python | **START HERE** in exam — all methods in one file, swap f(x) and run |
| [`basic/01_errors_and_approximations.py`](basic/01_errors_and_approximations.py) | 🐍 Python | Round-off, truncation, Taylor series, machine epsilon |
| [`basic/02_visualization_and_plotting.py`](basic/02_visualization_and_plotting.py) | 🐍 Python | All plot templates: standard, log scale, multi-root, convergence |
| [`basic/scanner.py`](basic/scanner.py) | 🐍 Python | **Incremental search scanner utility for root-finding intervals** |
| [`basic/07_error_tradeoff_plotter.py`](basic/07_error_tradeoff_plotter.py) | 🐍 Python | **Round-off vs. Truncation error tradeoff curve simulation & plot** |
| [`methods/03_bisection_and_false_position.py`](methods/03_bisection_and_false_position.py) | 🐍 Python | Bisection + False Position with full tables, comparison, multi-root |
| [`methods/04_newton_raphson.py`](methods/04_newton_raphson.py) | 🐍 Python | Newton-Raphson with failure demos, dipstick, diode, convergence plot |
| [`methods/05_bairstow_method.py`](methods/05_bairstow_method.py) | 🐍 Python | Bairstow's method — all roots of polynomials, step-by-step tables |
| [`prev_year_solves/06_prev_batch_all.py`](prev_year_solves/06_prev_batch_all.py) | 🐍 Python | **Previous batch online exam questions (A2, B2, A1) with exact formulations** |
| [`prev-section/`](prev-section/) | 📁 Folder | **Modular solutions for previous years' sections (A1, B1, B2, C1/C2)** |
| [`docs/STUDY_GUIDE.md`](docs/STUDY_GUIDE.md) | 📖 Docs | **Full study guide** — formulas, tables, exam tips, method comparison |
| [`docs/NEWTON_RAPHSON_DEEP_DIVE.md`](docs/NEWTON_RAPHSON_DEEP_DIVE.md) | 📖 Docs | NR geometry proof, quadratic convergence, all 5 failure modes |
| [`docs/BAIRSTOW_DEEP_DIVE.md`](docs/BAIRSTOW_DEEP_DIVE.md) | 📖 Docs | Bairstow algorithm derivation with worked numeric example |

---

### 🚀 Exam Strategy

#### Step 1: Identify the question type
- **Plot only** → use `basic/02_visualization_and_plotting.py` templates
- **Bisection / False Position** → copy from `methods/03_bisection_and_false_position.py`  
- **Newton-Raphson** → copy from `methods/04_newton_raphson.py`, derive `df(x)` first
- **Bairstow** → copy from `methods/05_bairstow_method.py`
- **Unsure** → open `cheatsheets/00_exam_cheatsheet.py` and work from there

#### Step 2: The Universal Workflow
```
1. Define f(x) [and df(x) for NR]
2. Plot f(x) — ALWAYS FIRST, marks deducted if skipped
3. Scan for sign-change intervals
4. Verify: f(xl)·f(xu) < 0
5. Run the method
6. Print the iteration table
7. State the root clearly
8. Save the graph: plt.savefig('answer.png', dpi=150)
```

#### Step 3: Check tolerances
- `tol=0.0001` means **0.0001%** (4 significant figures per Scarborough)
- `tol=0.001` means **0.001%** (3 significant figures)
- The error formula always multiplies by 100: `ea = |...| * 100`

---

### 📐 Formula Summary Card

```
┌──────────────────────────────────────────────────────────────────────┐
│ BISECTION:       xr = (xl + xu) / 2                                  │
│ FALSE POSITION:  xr = xu - f(xu)·(xl-xu) / (f(xl)-f(xu))           │
│ NEWTON-RAPHSON:  x_{i+1} = x_i - f(x_i) / f'(x_i)                  │
│                                                                       │
│ ERROR:           ea = |x_new - x_old| / |x_new| × 100%              │
│ SCARBOROUGH:     tol_s = 0.5 × 10^(2-n)  % for n sig figs           │
│                                                                       │
│ UPDATE RULE (Bisection & FP):                                        │
│   if f(xl)·f(xr) < 0  →  xu = xr                                    │
│   else                →  xl = xr                                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

### ⚠️ Critical Python Traps

1. **Cube root:** `np.cbrt(x)` NOT `x**(1/3)` — the latter gives complex numbers for negative x
2. **Log in NumPy:** Use `np.log(x)` (natural log), NOT `math.log(x)` with numpy arrays
3. **plt.show()** blocks code — if plotting slows you down, comment out `plt.show()` and keep only `plt.savefig()`
4. **Tolerance units:** `ea` is already in `%`. Compare `ea <= 0.0001`, not `ea <= 0.000001`

---

### 📝 NR Failure Modes (Must Memorize)

| Trap | Function | x₀ | Symptom | Fix |
|------|----------|----|---------|-----|
| 2-Cycle | x³−2x+2 | 0 | Infinite 0→1→0→1 loop | x₀ = −1.5 |
| Zero deriv | sin(x) | π/2 | Division by zero | Shift x₀ |
| Diverge | ∛x | any | Doubles each step | Use bisection |
| Overshoot | arctan(x) | 1.5 | Shoots to ±∞ | Plot first, choose near root |
| Slow | (x−2)³ | any | Linear convergence | Modified NR: x_{i+1} = x_i − m·f/f' |

---

### 🔬 Bairstow Quick Reference

```
Coefficients: HIGHEST POWER FIRST
  x⁴ − 5x³ + 7x² − 5x + 6  →  [1, -5, 7, -5, 6]

b-array:  b₀=a₀,  b₁=a₁+r·b₀,  bᵢ=aᵢ+r·bᵢ₋₁+s·bᵢ₋₂
c-array:  same formula applied to b[0..n-1] (drop bₙ!)

Corrections:   D = c[n-2]² - c[n-3]·c[n-1]
               Δr = (-b[n-1]·c[n-2] + b[n]·c[n-3]) / D
               Δs = (-b[n]·c[n-2]   + b[n-1]·c[n-1]) / D

Root extract:  disc = r² + 4s
               If disc ≥ 0: roots = (r ± √disc) / 2  (real)
               If disc < 0: roots = r/2 ± i·√|disc|/2  (complex)

Deflate:       next polynomial = b[0..n-2]
```
