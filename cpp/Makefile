CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter

BIN = bin
all: $(BIN)/uhucpm_mine

$(BIN)/uhucpm_mine: uhucpm_mine.cpp miner.cpp miner.hpp io.hpp
	mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ uhucpm_mine.cpp miner.cpp

clean:
	rm -rf $(BIN)

all: $(BIN)/pso_mucp
$(BIN)/pso_mucp: pso_mucp.cpp io.hpp
	mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ pso_mucp.cpp

all: $(BIN)/post_counts
$(BIN)/post_counts: post_counts.cpp ceb.cpp ceb.hpp io.hpp
	mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ post_counts.cpp ceb.cpp

all: $(BIN)/fsq_prep
$(BIN)/fsq_prep: fsq_prep.cpp utm.hpp
	mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ fsq_prep.cpp
