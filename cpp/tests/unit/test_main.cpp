// Runs every registered unit test. Usage: unit_tests [substring]  (only the tests whose name contains the substring)
#include <iostream>
#include <string>

#include "test_framework.hpp"

int main(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : "";
    int run = 0, failedTests = 0;
    for (const auto& t : unit::registry()) {
        if (!filter.empty() && t.name.find(filter) == std::string::npos) continue;
        const int before = unit::failureCount();
        try {
            t.body();
        } catch (const std::exception& e) {
            unit::fail(__FILE__, __LINE__, std::string("uncaught exception: ") + e.what());
        }
        ++run;
        const bool ok = unit::failureCount() == before;
        if (!ok) ++failedTests;
        std::cout << (ok ? "ok      " : "FAILED  ") << t.name << "\n";
    }
    std::cout << run << " tests, " << failedTests << " failed, " << unit::failureCount() << " failed checks\n";
    return failedTests ? 1 : 0;
}
