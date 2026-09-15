# Research papers (LaTeX)

## English
```bash
cd papers/en
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```
Output: `papers/en/main.pdf`

## Arabic (requires XeLaTeX + Arabic Typesetting font)
```bash
cd papers/ar
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```
Output: `papers/ar/main.pdf`

Figures are loaded from `experiments/figures/`.
