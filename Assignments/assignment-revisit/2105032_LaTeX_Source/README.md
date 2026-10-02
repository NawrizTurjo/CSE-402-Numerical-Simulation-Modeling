# Gradient Methods · CSE 402 Assignment · Roll 2105032

LaTeX Beamer teaching module for **Topic 3: Gradient Methods**.

- **How it works (sections 1–4):** fitting a line, the gradient and stationary points, the update rule (Taylor and forward-Euler views), and watching GD move on the SSR contours and in 3D.
- **When it works, and when it fails (sections 5–7):** the learning rate and stability, order of convergence and guarantees, the condition number and failure modes.
- **Fixes and scale (sections 8–9):** schedules and backtracking, momentum, Nesterov and Adam; SGD, its noise floor, automatic differentiation and the edge of stability.
- **Part B:** 10 problems (M1–M5 mathematical, A1–A5 applications). Each one is followed immediately by its complete solution.

## Build

Requires TeX Live or MiKTeX with `beamer`, `metropolis`, `fira`, `newtxsf`, `tikz`, `tcolorbox`, `listings`, `booktabs` and `etoolbox`. No shell-escape and no Python are needed, because all figures are pre-generated.

```
pdflatex 2105032.tex
bibtex   2105032
pdflatex 2105032.tex
pdflatex 2105032.tex
```

`2105032.bbl` is included, so the deck also builds without BibTeX.

## Files

| Path | Contents |
|---|---|
| `2105032.tex` | primary document |
| `preamble.tex` | theme (paper-navy, built on Metropolis) |
| `sections/01_why.tex` … `10_summary.tex` | the nine sections and the summary |
| `sections/problems/` | Part B: overview, M1–M5, A1–A5 |
| `figures/generated/` | figures (vector PDF) made by `scripts/make_figures.py` |
| `figures/tikz/` | TikZ diagrams |
| `scripts/verify_problems.py` | recomputes every number in the solutions, with assertions |
| `bibliography/references.bib` | the 20 references cited in the deck |
