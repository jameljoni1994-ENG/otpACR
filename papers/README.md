# Research papers (LaTeX)

Figures load from `experiments/figures/` (including phase-10: `25`–`27`).
Results referenced in the papers include early stopping (phase~9) and full MNIST / $\ell_2$ path / sketch scaling (phase~10).

## English
```bash
cd papers/en
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```
Output: `papers/en/main.pdf`

## Arabic (requires XeLaTeX + Arabic-capable font, e.g. Segoe UI)
```bash
cd papers/ar
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```
Output: `papers/ar/main.pdf`
