# Thesis sources

LaTeX sources for the thesis text.

- `main.tex`: preamble and document skeleton. Each chapter is pulled in with `\input`.
- `Chapters/`: one file per chapter (`00Abstract.tex` to `10Appendix.tex`).
- `Tables/`: tables included from the chapters.
- `references.bib`: bibliography, processed with biblatex and biber.

Figures are not included in this repository. The chapters reference images under
`Figures/<topic>/...`. Through `\graphicspath` they resolve either from this directory or from the repository root. Most of them are produced by the scripts in
`../code/data_visualization/` and `../code/validation/`, which need the survey data. They are
not distributed here.

Build with figures present:

```sh
latexmk -pdf main.tex
```
