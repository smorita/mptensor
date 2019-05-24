PDF=manual_doxygen.pdf

.PHONY: all doc pdf clean

all: doc

doc:
	doxygen Doxyfile

pdf: doc
	make -C latex
	cp latex/refman.pdf $(PDF)

clean:
	rm -rf html latex $(PDF)
