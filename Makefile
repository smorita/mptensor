CXX=mpicxx
CXXFLAGS=-O3
LDFLAGS=-L/opt/local/lib -lscalapack

-include Makefile.option

.PHONY: all mptensor tests clean

all: mptensor

mptensor:
	$(MAKE) -C src

tests:
	$(MAKE) -C tests

clean:
	$(MAKE) -C src clean
	$(MAKE) -C tests clean
