CXX			= mpicxx
CFLAGS 		= -Wall -openmp
TARGET 		= main


INCLUDE_DIR = include
SRC_DIR		= source
OBJ_DIR		= build/obj
BIN_DIR		= build/bin

HEADERS 	= $(wildcard $(INCLUDE_DIR)/*.h)
SOURCES		= $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS 	= $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o, $(SOURCES))
EXECUTABLE 	= $(BIN_DIR)/$(TARGET)

.PHONY: all $(TARGET) test report clean

all:		$(TARGET)

$(TARGET): $(EXECUTABLE)

$(EXECUTABLE):	$(OBJECTS)
		$(CXX) $(CFLAGS) -o $(BIN_DIR)/$(TARGET) $(OBJECTS) $(LINKFLAGS)

$(OBJ_DIR)/%.o:	$(SRC_DIR)/%.cpp $(HEADERS)
		mkdir -p build
			mkdir -p $(BIN_DIR) $(OBJ_DIR)
				$(CXX) -I $(INCLUDE_DIR)  $(CFLAGS) -c -o $@ $<


clean:
		rm -Rf build

