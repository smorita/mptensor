-include ../../Makefile.option

.PHONY: all clean depend
.SUFFIXES: .cc .hpp .o .out
SRC=$(shell ls *.cc)
HED=$(shell ls *.hpp)
OBJ=$(SRC:.cc=.o)
TARGET=$(SRC:.cc=.out)

MPTENSOR_DIR = ../../src
MPTENSOR = -I$(MPTENSOR_DIR) -L$(MPTENSOR_DIR) -lmptensor


all: $(TARGET)

%.out : %.cc
	$(CXX) -o $@ $< $(MPTENSOR) $(LDFLAGS) $(CXXFLAGS)

clean:
	@rm -vf *.out *.o
	@rm -vf Makefile.depend

depend Makefile.depend:
	g++ -MM -MG $(SRC) > Makefile.depend

-include Makefile.depend
