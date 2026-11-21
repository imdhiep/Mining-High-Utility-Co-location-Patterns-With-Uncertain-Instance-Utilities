#include "common/process_stats.hpp"

#include <cstdio>
#include <sys/resource.h>

namespace colocation {

// VmHWM of /proc/self/status: the peak resident set size of this process image only. getrusage(RUSAGE_SELF) is not
// used because on Linux its ru_maxrss keeps the resident size of the parent at fork across exec, so a program started
// from a large process (the Python harness) would report the parent's memory.
long long peakMemoryKb() {
    long long kb = -1;
    if (std::FILE* f = std::fopen("/proc/self/status", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f))
            if (std::sscanf(line, "VmHWM: %lld", &kb) == 1) break;
        std::fclose(f);
    }
    if (kb >= 0) return kb;
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

}  // namespace colocation
