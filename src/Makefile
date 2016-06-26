TARGET=libmptensor.a

-include ../Makefile.option

.PHONY: all clean depend
.SUFFIXES: .cc .hpp .o
SRC=$(shell find mptensor -name "*.cc")
HED=$(shell find mptensor -name "*.hpp")
OBJ=$(SRC:.cc=.o)

all: $(TARGET)

$(TARGET) : $(OBJ) $(HED)
	$(AR) rvu $(TARGET) $(OBJ) $(RANDOM_OBJ)

.cc.o:
	$(CXX) -c $< -o $@ $(LDFLAGS) $(CXXFLAGS)

clean:
	@rm -vf $(TARGET)
	@rm -vf mptensor/*.o mptensor/*/*.o
	@rm -vf Makefile.depend

depend Makefile.depend:
	g++ -MM -MG $(SRC) > Makefile.depend


-include Makefile.depend
