# Build the P1 draft. Run from paper/p1/:  make            (numbers, tables, figures, pdf)
# The TMLR style files stay untouched in tmlr_template/; TeX finds them via TEXINPUTS/BSTINPUTS.
PY      ?= ../../.venv/bin/python
# Fixed PDF dates (UTC, 2026-01-01): the submitted PDF carries no build time or local timezone.
export SOURCE_DATE_EPOCH ?= 1767225600
export FORCE_SOURCE_DATE ?= 1
TEXENV  = TEXINPUTS=./tmlr_template:$${TEXINPUTS} BSTINPUTS=./tmlr_template:$${BSTINPUTS}

.PHONY: all numbers tables figures pdf pdf-texlive clean

all: numbers tables figures pdf

numbers:
	$(PY) make_numbers.py

tables:
	$(PY) make_tables.py

figures:
	$(PY) figures/make_figures.py

# Default: tectonic (brew install tectonic); output in build/.
pdf: main.tex numbers_p1.tex tables_p1.tex tables_numbers_p1.tex refs.bib
	mkdir -p build
	tectonic -X compile main.tex -Z search-path=tmlr_template --outdir build --keep-logs --keep-intermediates

# Alternative with a TeX Live installation.
pdf-texlive: main.tex numbers_p1.tex tables_p1.tex tables_numbers_p1.tex refs.bib
	$(TEXENV) pdflatex -interaction=nonstopmode main.tex
	$(TEXENV) bibtex main
	$(TEXENV) pdflatex -interaction=nonstopmode main.tex
	$(TEXENV) pdflatex -interaction=nonstopmode main.tex

clean:
	rm -rf build main.aux main.bbl main.blg main.log main.out main.pdf
