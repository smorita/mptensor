-include ../../Makefile.option

.PHONY: all clean depend
.SUFFIXES: .cc .hpp .o .out

TARGET = trg.out hotrg.out
SRC = ising.cc trg.cc hotrg.cc
OBJS = ising.o

MPTENSOR_DIR = ../../src
MPTENSOR = -I$(MPTENSOR_DIR) -L$(MPTENSOR_DIR) -lmptensor


all: $(TARGET)

%.out : %.o $(OBJS)
	$(CXX) -o $@ $< $(OBJS) $(MPTENSOR) $(LDFLAGS) $(CXXFLAGS)

%.o : %.cc
	$(CXX) -c $< $(MPTENSOR) $(LDFLAGS) $(CXXFLAGS)

clean:
	rm -vf *.out *.o

depend Makefile.depend:
	g++ -MM -MG $(SRC) > Makefile.depend

-include Makefile.depend
