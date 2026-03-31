MAIN = main
LATEX = pdflatex
BIB = bibtex

.PHONY: pdf clean

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib neurips_2026.sty
	$(LATEX) $(MAIN)
	$(BIB) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc
	rm -f $(MAIN).synctex.gz $(MAIN).fls $(MAIN).fdb_latexmk
