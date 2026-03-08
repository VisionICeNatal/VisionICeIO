# Makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two:
#   SPHINXOPTS    ?= -W --keep-going -n
#   SPHINXBUILD   ?= sphinx-build
#   PAPER         ?= a4

# Internal variables
SPHINXOPTS    ?= -W --keep-going -n
SPHINXBUILD   ?= sphinx-build
PAPER         ?= a4
BUILDDIR      ?= _build

.PHONY: help Makefile serve

# Put it first so that "make" without argument is like "make help"
help:
	@$(SPHINXBUILD) -M help "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*

serve:
	@$(SPHINXBUILD) -b html "$(CURDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "Serving docs at http://localhost:8000 — press Ctrl+C to stop."
	@python -m http.server 8000 --directory "$(BUILDDIR)/html"

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(CURDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
