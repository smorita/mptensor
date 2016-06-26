TARGET=tensor_test.out

-include ../Makefile.option

.PHONY: all lib run test clean depend
.SUFFIXES: .cc .hpp .o
SRC=$(shell ls *.cc)
HED=$(shell ls *.hpp)
OBJ=$(SRC:.cc=.o)

LIB=../src/libmptensor.a
CXXFLAGS += -I../src

all: lib $(TARGET)

$(TARGET): $(OBJ) $(HED) $(LIB)
	$(CXX) -o $(TARGET) $(OBJ) $(LIB) $(LDFLAGS) $(CXXFLAGS)

$(LIB) lib:
	$(MAKE) -C ../src

.cc.o:
	$(CXX) -c $< $(LDFLAGS) $(CXXFLAGS)

run test:
	mpirun -np 4 ./$(TARGET) 10

clean:
	@rm -vf *.out *.o
	@rm -vf Makefile.depend

depend Makefile.depend:
	g++ -MM -MG $(SRC) > Makefile.depend

-include Makefile.depend
