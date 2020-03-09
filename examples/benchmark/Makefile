-include ../../Makefile.option

.PHONY: all clean depend
.SUFFIXES: .cc .hpp .o .out
SRC=$(shell ls *.cc)
HED=$(shell ls *.hpp)
OBJ=$(SRC:.cc=.o)
TARGET=$(SRC:.cc=.out)

MPTENSOR_SOURCE_DIR = ../../src
MPTENSOR_INCLUDE_DIR = ../../include
MPTENSOR_FLAGS = -I$(MPTENSOR_INCLUDE_DIR) -L$(MPTENSOR_SOURCE_DIR) -lmptensor


all: $(TARGET)

%.out : %.cc
	$(CXX) -o $@ $< $(MPTENSOR_FLAGS) $(LDFLAGS) $(CXXFLAGS)

clean:
	@rm -vf *.out *.o
	@rm -vf Makefile.depend

depend Makefile.depend:
	g++ -MM -MG -I$(MPTENSOR_FLAGS) $(SRC) > Makefile.depend

-include Makefile.depend
