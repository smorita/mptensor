-include ../../Makefile.option

.PHONY: all clean depend
.SUFFIXES: .cc .hpp .o .out

TARGET = trg.out hotrg.out atrg.out
SRC = ising.cc trg.cc hotrg.cc atrg.cc
OBJS = ising.o

MPTENSOR_SOURCE_DIR = ../../src
MPTENSOR_INCLUDE_DIR = ../../include
MPTENSOR_FLAGS = -I$(MPTENSOR_INCLUDE_DIR) -L$(MPTENSOR_SOURCE_DIR) -lmptensor


all: $(TARGET)

%.out : %.o $(OBJS)
	$(CXX) -o $@ $< $(OBJS) $(MPTENSOR_FLAGS) $(LDFLAGS) $(CXXFLAGS)

%.o : %.cc
	$(CXX) -c $< $(MPTENSOR_FLAGS) $(LDFLAGS) $(CXXFLAGS)

clean:
	rm -vf *.out *.o

depend Makefile.depend:
	g++ -MM -MG $(MPTENSOR_FLAGS) $(SRC) > Makefile.depend

-include Makefile.depend
