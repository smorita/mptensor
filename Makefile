CXX=mpicxx
CXXFLAGS=-O3
LDFLAGS=-L/opt/local/lib -lscalapack

-include Makefile.option

.PHONY: all mptensor tests clean doc doxygen

all: mptensor

mptensor:
	$(MAKE) -C src

tests:
	$(MAKE) -C tests

doc doxygen:
	doxygen Doxyfile

clean:
	$(MAKE) -C src clean
	$(MAKE) -C tests clean
	rm -rf doxygen_docs
