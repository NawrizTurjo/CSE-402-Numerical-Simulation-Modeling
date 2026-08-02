# Master Template Quick Reference (`template.py` / `P`)

> **Usage in solution scripts:** `from common import P, use, banner`

---

## 1. Setup & Registration

### `use(f, df=None)`
- **What it does**: Registers your problem's $f(x)$ and $df(x)$ with master solvers.
- **Arguments**: `f` (callable), `df` (callable, optional for Newton-Raphson).
- **Returns**: `None`.

---

## 2. Scanning & Domain Safety

### `safe_eval(x, func=f)`
- **What it does**: Evaluates `func(x)`, catching domain errors ($\ln(\le 0)$, div-by-zero, overflow).
- **Arguments**: `x` (float), `func` (callable, default: global `f`).
- **Returns**: `(value, None)` on success, `(None, error_str)` on failure.

### `coarse_scan(a, b, step=0.1, direct_tol=1e-8)`
- **What it does**: Scans $[a, b]$ in steps, classifying points into direct hits ($|f| \le 10^{-8}$) and sign-change brackets.
- **Arguments**: `a` (float), `b` (float), `step` (float), `direct_tol` (float).
- **Returns**: `(direct_candidates, interval_candidates)` tuple of lists.

### `classify_and_verify(candidate, kind, extra_filter=None, residual_tol=1e-6)`
- **What it does**: Verifies if $|f(\text{candidate})| \le \text{residual\_tol}$ and returns an ACCEPT/REJECT verdict.
- **Arguments**: `candidate` (float), `kind` (str: `'interval'`, `'bisection'`, etc.), `extra_filter` (callable), `residual_tol` (float).
- **Returns**: Dict `{'type', 'x', 'f_x', 'error', 'accepted'}`.

### `find_all_roots(a, b, step=0.1, method='bisection', tol=0.0001, m=1, getInterval=False)`
- **What it does**: Automatically scans $[a, b]$ and solves all sign-change brackets.
- **Arguments**: `a`, `b`, `step`, `method` (`'bisection'`, `'false_position'`, `'false_position_illinois'`, `'newton_raphson'`), `tol`, `m`, `getInterval` (bool).
- **Returns**: `roots` list, or `(roots, intervals)` if `getInterval=True`.

---

## 3. Root-Finding Solvers

### `bisection(xl, xu, tol=0.0001, max_iter=200, verbose=True)`
- **What it does**: Midpoint bracketing solver ($x_r = \frac{x_l + x_u}{2}$).
- **Arguments**: `xl` (float), `xu` (float), `tol` (float), `max_iter` (int), `verbose` (bool).
- **Returns**: `(root, i, ea, xl_updates, xu_updates, history)` or `None`.

### `false_position(xl, xu, tol=0.0001, max_iter=500, verbose=True)`
- **What it does**: Standard linear secant interpolation solver.
- **Arguments**: `xl`, `xu`, `tol`, `max_iter`, `verbose`.
- **Returns**: `(root, i, ea, xl_updates, xu_updates, history)` or `None`.

### `false_position_opt(xl, xu, tol=0.0001, max_iter=500, verbose=True)`
- **What it does**: Optimized secant solver reusing precomputed $f(x_l)$/$f(x_u)$ with zero-denom guard.
- **Arguments**: `xl`, `xu`, `tol`, `max_iter`, `verbose`.
- **Returns**: `(root, i, ea, xl_updates, xu_updates, history)` or `None`.

### `false_position_illinois(xl, xu, tol=0.0001, max_iter=500, verbose=True)`
- **What it does**: Anti-stagnation secant solver (halves weight on stagnant bound).
- **Arguments**: `xl`, `xu`, `tol`, `max_iter`, `verbose`.
- **Returns**: `(root, i, ea, xl_updates, xu_updates, history)` or `None`.

### `_newton_raphson_core(x0, m=1, tol=0.0001, max_iter=100, verbose=True)`
- **What it does**: Tangent update core solver ($x_{i+1} = x_i - m \frac{f(x_i)}{f'(x_i)}$) with cycle detection.
- **Arguments**: `x0` (float), `m` (multiplicity, default 1), `tol`, `max_iter`, `verbose`.
- **Returns**: `(root, history)` tuple where history is list of dicts.

### `newton_raphson(x0, m=1, tol=0.0001, max_iter=100, verbose=True)`
- **What it does**: Newton-Raphson solver reading global `f` and `df`.
- **Arguments**: `x0`, `m`, `tol`, `max_iter`, `verbose`.
- **Returns**: `root` float (or `None`).

### `newton_raphson_numeric(x0, m=1, h=1e-6, tol=0.0001, max_iter=100, verbose=True)`
- **What it does**: Newton-Raphson using central difference numeric derivative.
- **Arguments**: `x0`, `m`, `h` (step size), `tol`, `max_iter`, `verbose`.
- **Returns**: `root` float (or `None`).

### `compare_methods(xl, xu, tol=0.0001, m=1)`
- **What it does**: Runs Bisection, False Position, and Newton-Raphson silently and prints side-by-side comparison table.
- **Arguments**: `xl`, `xu`, `tol`, `m`.
- **Returns**: `(bi, fp, nr)` 3-tuple.

---

## 4. Visualization & Plotting

### `plot_standard(f_np, a, b, title='f(x)', fname='fig/graph.png')`
- **What it does**: Standard function curve with zero line.
- **Arguments**: `f_np` (vectorized callable), `a`, `b`, `title`, `fname`.
- **Returns**: `None` (saves figure to `fig/`).

### `plot_logscale(f_np, a, b, title='f(x) log scale', fname='fig/graph_log.png')`
- **What it does**: Logarithmic x-axis plot for wide multi-order domains ($10^{-4}$ to $10^4$).
- **Arguments**: `f_np`, `a`, `b`, `title`, `fname`.
- **Returns**: `None` (saves figure to `fig/`).

### `plot_with_root(f_np, a, b, root, xl=None, xu=None, title='Root Found', fname='fig/graph_root.png')`
- **What it does**: Plots function curve, optional bracket shading, and root marker.
- **Arguments**: `f_np`, `a`, `b`, `root`, `xl`, `xu`, `title`, `fname`.
- **Returns**: `None` (saves figure to `fig/`).

### `plot_with_brackets(f_np, a, b, intervals=None, roots=None, title='Multi-Root Finding', fname='fig/graph_multi.png')`
- **What it does**: Shades all scanned sign-change brackets and plots root markers.
- **Arguments**: `f_np`, `a`, `b`, `intervals` (list of tuples), `roots` (list of floats), `title`, `fname`.
- **Returns**: `None` (saves figure to `fig/`).

### `plot_convergence(errors, title='Convergence of ea', fname='fig/convergence.png')`
- **What it does**: Semilog plot of relative error $\varepsilon_a$ vs iteration.
- **Arguments**: `errors` (list of floats OR `history` dict list), `title`, `fname`.
- **Returns**: `None` (saves figure to `fig/`).

### `plot_newton_raphson(x0, a, b, m=1, tol=0.0001, title='Tangent Geometry', fname='fig/nr_tangents.png')`
- **What it does**: Visualizes Newton-Raphson tangent line geometry per iteration.
- **Arguments**: `x0`, `a`, `b`, `m`, `tol`, `title`, `fname`.
- **Returns**: `None` (saves figure to `fig/`).

---

## 5. Error Analysis & Theory Helpers

### `scarborough_tolerance(n_sig_figs)`
- **What it does**: Calculates Scarborough tolerance $\varepsilon_s = 0.5 \times 10^{2-n} \%$.
- **Arguments**: `n_sig_figs` (int).
- **Returns**: `es` float.

### `calc_sig_digit(err)`
- **What it does**: Calculates guaranteed significant digits $n = \lfloor 2 - \log_{10}(2 \cdot \text{err}) \rfloor$.
- **Arguments**: `err` (float).
- **Returns**: `sig_digits` int.

### `calculate_m_sig_tol(m)`
- **What it does**: Shortcut for Scarborough tolerance $0.5 \times 10^{2-m} \%$.
- **Arguments**: `m` (int).
- **Returns**: `tol` float.

### `compute_machine_epsilon()`
- **What it does**: Computes 64-bit float machine precision limit ($\approx 2.22 \times 10^{-16}$).
- **Arguments**: None.
- **Returns**: `eps` float.

### `forward_diff(f_func, x, h)` / `backward_diff` / `central_diff`
- **What it does**: Evaluates $O(h)$ forward/backward or $O(h^2)$ central difference derivative.
- **Arguments**: `f_func` (callable), `x` (float), `h` (float).
- **Returns**: `approx_derivative` float.

### `diff_truncation_table(f_func, fprime_exact, x0, h_list)`
- **What it does**: Prints table comparing Forward, Backward, Central diffs against exact $f'(x_0)$.
- **Arguments**: `f_func`, `fprime_exact`, `x0`, `h_list` (list of floats).
- **Returns**: `None`.

### `maclaurin_series(term_recurrence, x, first_term=1.0, sig_digits=3)`
- **What it does**: Solves any Maclaurin/Taylor series with Scarborough stopping criterion.
- **Arguments**: `term_recurrence` (`lambda prev, n, x: ...`), `x`, `first_term`, `sig_digits`.
- **Returns**: `(sum_val, history)` tuple.

### `riemann_sum(f_func, a, b, n)`
- **What it does**: Left-endpoint Riemann sum approximation $\sum f(x_i) \Delta x$.
- **Arguments**: `f_func` (callable), `a` (float), `b` (float), `n` (int).
- **Returns**: `integral_approx` float.

### `demo_error_tradeoff(x_target=1.0, fname='fig/error_tradeoff.png')`
- **What it does**: Finds and plots optimal step size $h$ on V-curve balancing round-off vs truncation.
- **Arguments**: `x_target` (float), `fname` (str).
- **Returns**: `(optimal_h, optimal_error)` tuple.
