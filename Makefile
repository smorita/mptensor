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
	$(MAKE) -C doc/doxygen

clean:
	$(MAKE) -C src clean
	$(MAKE) -C tests clean
	$(MAKE) -C doc/doxygen clean
	rm -rf doxygen_docs
