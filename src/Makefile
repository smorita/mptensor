TARGET=libmptensor.a

-include ../Makefile.option
MPTENSOR_INCLUDE_DIR=../include
CXXFLAGS += -I$(MPTENSOR_INCLUDE_DIR)

.PHONY: all clean depend
.SUFFIXES: .cc .hpp .o
SRC=$(shell find . -name "*.cc")
# HED=$(shell find mptensor -name "*.hpp")
OBJ=$(SRC:.cc=.o)

all: $(TARGET)

$(TARGET) : $(OBJ)
	$(AR) rvu $(TARGET) $(OBJ) $(RANDOM_OBJ)

.cc.o:
	$(CXX) -c $< -o $@ $(LDFLAGS) $(CXXFLAGS)

clean:
	@rm -vf $(TARGET)
	@rm -vf *.o */*.o
	@rm -vf Makefile.depend

depend Makefile.depend:
	g++ -MM -MG -I$(MPTENSOR_INCLUDE_DIR) $(SRC) > Makefile.depend


-include Makefile.depend
