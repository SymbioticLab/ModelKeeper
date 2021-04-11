CC=g++
CFLAGS=-c -O2 -Wall
LDFLAGS=
SOURCES=src/DMSTNode.cpp src/Simulator.cpp src/Network.cpp src/main.cpp
OBJECTS=$(patsubst src/%.cpp, obj/%.o, $(SOURCES))
EXECUTABLE=bin/MST

$(shell [ -d bin ] || mkdir -p bin)
$(shell [ -d obj ] || mkdir -p obj)

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

obj/%.o: src/%.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)
