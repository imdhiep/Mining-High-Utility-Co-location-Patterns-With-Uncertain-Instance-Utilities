// Wall-clock timer and peak resident memory of the current process.
#pragma once

#include <chrono>

namespace colocation {

class Stopwatch {
public:
    Stopwatch() : start_(std::chrono::steady_clock::now()) {}
    double seconds() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count(); }

private:
    std::chrono::steady_clock::time_point start_;
};

// peak resident set size in kilobytes (getrusage)
long long peakMemoryKb();

}  // namespace colocation
