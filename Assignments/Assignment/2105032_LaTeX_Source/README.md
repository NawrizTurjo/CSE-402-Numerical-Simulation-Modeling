# Gradient Methods — CSE 402 Assignment (Roll 2105032)

LaTeX Beamer teaching module for **Topic 3: Gradient Methods**.

- **Part A:** concept presentation and advanced theory.
- **Part B:** 10 problems (M1–M5 mathematical, A1–A5 applications), each followed immediately by its complete solution.

## Build

Requires a standard TeX distribution (TeX Live or MiKTeX) with `beamer`, the `metropolis` theme, `tikz`, `tcolorbox`, `listings`, `booktabs`, `etoolbox`, and the fonts `fira` (Fira Sans and Fira Mono) and `newtxsf` (sans-serif maths). All of these ship with TeX Live full and are installed on demand by MiKTeX. No shell-escape and no Python are needed: all figures are pre-generated.

```
pdflatex 2105032.tex
bibtex   2105032
pdflatex 2105032.tex
pdflatex 2105032.tex
```

`latexmk -pdf 2105032.tex` does the same. The output is `2105032.pdf`.

The archive also contains `2105032.bbl` (the pre-built bibliography). If BibTeX is unavailable, running `pdflatex` twice still produces the complete deck.

## File map

| Path | Contents |
|---|---|
| `2105032.tex` | Primary document: front matter, section order, references |
| `preamble.tex` | Packages, Metropolis setup, colours, boxes, maths macros |
| `sections/01_…14_*.tex` | Part A theory modules (one file per section) |
| `sections/problems/intro.tex` | Part B overview |
| `sections/problems/M1–M5.tex`, `A1–A5.tex` | Problem frame followed by its solution frames |
| `figures/generated/*.pdf` | Matplotlib vector figures (created by `scripts/make_figures.py`) |
| `figures/tikz/*.tikz` | TikZ diagrams (roadmap, steepest descent, descent lemma, sandwich, Euler disk, contour shapes, spring chain) |
| `bibliography/references.bib` | BibTeX database (20 verified references) |
| `scripts/make_figures.py` | Regenerates every figure in `figures/generated/` |
| `scripts/verify_problems.py` | Recomputes every number used in the Part B solutions, with assertions |
| `scripts/requirements.txt` | `numpy`, `matplotlib` |

## Reproducing the figures and checking the numbers (optional)

```
pip install -r scripts/requirements.txt
python scripts/make_figures.py
python scripts/verify_problems.py
```
