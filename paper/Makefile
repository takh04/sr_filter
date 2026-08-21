PYTHON ?= $(if $(wildcard ../.venv/bin/python),../.venv/bin/python,python3)
LATEXMK ?= latexmk
SOURCE_DATE_EPOCH ?= 1767225600
FORCE_SOURCE_DATE ?= 1
TZ := UTC

export SOURCE_DATE_EPOCH
export FORCE_SOURCE_DATE
export TZ

.PHONY: all figures static-figures online-figures online-gate pdf camera-ready check check-static check-page clean

all: pdf

static-figures:
	$(PYTHON) scripts/generate_static_figures.py

online-gate:
	$(PYTHON) scripts/generate_online_comparison.py --check-only

online-figures:
	$(PYTHON) scripts/generate_online_comparison.py

figures: static-figures online-figures

pdf: figures
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

camera-ready: pdf check

check-static:
	$(PYTHON) scripts/check_camera_ready.py

check-page:
	$(PYTHON) scripts/check_page_limit.py main.aux --max-pages 10

check: online-gate check-static check-page

clean:
	$(LATEXMK) -C main.tex
