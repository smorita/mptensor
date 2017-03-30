-include ../../Makefile.option

.PHONY: all clean depend
.SUFFIXES: .cc .hpp .o .out

TARGET = save_load.out

MPTENSOR_DIR = ../../src
MPTENSOR = -I$(MPTENSOR_DIR) -L$(MPTENSOR_DIR) -lmptensor


all: $(TARGET)

%.out : %.cc
	$(CXX) -o $@ $< $(MPTENSOR) $(LDFLAGS) $(CXXFLAGS)

clean:
	@rm -vf *.out *.o A.dat*
