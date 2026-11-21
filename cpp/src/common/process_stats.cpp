#include "common/process_stats.hpp"

#include <sys/resource.h>

namespace colocation {

long long peakMemoryKb() {
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

}  // namespace colocation
