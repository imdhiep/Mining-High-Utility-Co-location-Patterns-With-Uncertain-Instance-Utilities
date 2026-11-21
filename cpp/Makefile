# Build without CMake (same sources, flags and outputs as CMakeLists.txt).
#   make            programs in bin/
#   make test       unit tests (bin/unit_tests)
#   make clean
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
CPPFLAGS += -Isrc

BUILD = build
BIN = bin
APPS = uhucpm_mine pso_mucp post_counts fsq_prep yelp_prep

LIB_SRC = $(shell find src -name '*.cpp' | sort)
LIB_OBJ = $(LIB_SRC:%.cpp=$(BUILD)/%.o)
TEST_SRC = $(sort $(wildcard tests/unit/*.cpp))
TEST_OBJ = $(TEST_SRC:%.cpp=$(BUILD)/%.o)

all: $(APPS:%=$(BIN)/%)

$(BUILD)/libcolocation.a: $(LIB_OBJ)
	ar rcs $@ $^

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BIN)/%: $(BUILD)/apps/%.o $(BUILD)/libcolocation.a
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BIN)/unit_tests: CPPFLAGS += -Itests/unit
$(BIN)/unit_tests: $(TEST_OBJ) $(BUILD)/libcolocation.a
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $^

test: $(BIN)/unit_tests
	./$(BIN)/unit_tests

clean:
	rm -rf $(BUILD) $(BIN)

.PHONY: all test clean
.PRECIOUS: $(BUILD)/%.o
-include $(LIB_OBJ:.o=.d) $(TEST_OBJ:.o=.d) $(APPS:%=$(BUILD)/apps/%.d)
