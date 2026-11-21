// Minimal unit-test framework (no external dependency): TEST(name) registers a test, CHECK / CHECK_NEAR / CHECK_THROWS
// record failures without stopping the test, test_main.cpp runs every registered test and returns 1 on any failure.
#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace unit {

struct TestCase {
    std::string name;
    std::function<void()> body;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

inline int& failureCount() {
    static int failures = 0;
    return failures;
}

struct Registrar {
    Registrar(const std::string& name, std::function<void()> body) { registry().push_back({name, std::move(body)}); }
};

inline void fail(const char* file, int line, const std::string& message) {
    ++failureCount();
    std::cerr << "  FAILED " << file << ":" << line << ": " << message << "\n";
}

}  // namespace unit

#define UNIT_CONCAT_INNER(a, b) a##b
#define UNIT_CONCAT(a, b) UNIT_CONCAT_INNER(a, b)

#define TEST(name)                                                                       \
    static void UNIT_CONCAT(test_, name)();                                              \
    static unit::Registrar UNIT_CONCAT(registrar_, name)(#name, UNIT_CONCAT(test_, name)); \
    static void UNIT_CONCAT(test_, name)()

#define CHECK(condition)                                                          \
    do {                                                                          \
        if (!(condition)) unit::fail(__FILE__, __LINE__, "CHECK(" #condition ")"); \
    } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                                       \
    do {                                                                                              \
        const double a_ = (actual), e_ = (expected);                                                  \
        if (!(std::fabs(a_ - e_) <= (tolerance))) {                                                    \
            std::ostringstream m_;                                                                    \
            m_.precision(17);                                                                         \
            m_ << #actual << " = " << a_ << ", expected " << e_ << " (tolerance " << (tolerance) << ")"; \
            unit::fail(__FILE__, __LINE__, m_.str());                                                 \
        }                                                                                             \
    } while (0)

#define CHECK_THROWS(statement)                                                              \
    do {                                                                                     \
        bool thrown_ = false;                                                                \
        try {                                                                                \
            statement;                                                                       \
        } catch (...) {                                                                      \
            thrown_ = true;                                                                  \
        }                                                                                    \
        if (!thrown_) unit::fail(__FILE__, __LINE__, "expected an exception from " #statement); \
    } while (0)
