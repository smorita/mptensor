-include ../../Makefile.option

.PHONY: all clean depend
.SUFFIXES: .cc .hpp .o .out

TARGET = simple_example.out

MPTENSOR_SOURCE_DIR = ../../src
MPTENSOR_INCLUDE_DIR = ../../include
MPTENSOR_FLAGS = -I$(MPTENSOR_INCLUDE_DIR) -L$(MPTENSOR_SOURCE_DIR) -lmptensor


all: $(TARGET)

%.out : %.cc
	$(CXX) -o $@ $< $(MPTENSOR_FLAGS) $(LDFLAGS) $(CXXFLAGS)

clean:
	@rm -vf *.out *.o
