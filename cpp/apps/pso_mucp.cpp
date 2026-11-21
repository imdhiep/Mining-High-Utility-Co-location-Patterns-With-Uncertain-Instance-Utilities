// One PSO-MUCP run (C++ port of the authors' code, see src/baselines/psomucp_miner.hpp); prints one JSON line.
// Usage: pso_mucp DATA_CSV UTILITY MIN_UTILITY D [--norm NORM]
//   UTILITY  "A:2,B:-1,..." or a Python dict literal "{'A': 2, 'B': -1}"
//   --norm   the normalizer ALL; default as in the authors' code: sum over features of count x utility (all signs)
#include <iostream>
#include <stdexcept>

#include "baselines/psomucp_miner.hpp"
#include "common/command_line.hpp"
#include "common/csv_table.hpp"
#include "common/json_writer.hpp"
#include "common/process_stats.hpp"

using namespace colocation;

namespace {

constexpr int kExitUsage = 2;

int run(int argc, char** argv) {
    const Stopwatch clock;
    const CommandLine cl(argc, argv);
    if (cl.positionalCount() < 4) {
        std::cerr << "usage: pso_mucp DATA_CSV UTILITY MIN_UTILITY D [--norm NORM]\n";
        return kExitUsage;
    }
    PsoMucpConfig config;
    config.utility = PsoMucpConfig::parseUtility(cl.positional(1));
    config.minUtility = cl.positionalDouble(2);
    config.distance = cl.positionalDouble(3);
    if (cl.has("norm")) config.normalizer = cl.option("norm", 0.0);
    PsoMucpMiner miner(PsoMucpMiner::loadInstances(CsvTable::read(cl.positional(0))), config);
    const PsoMucpResult r = miner.mine();
    JsonLineWriter out;
    out.field("mode", "psomucp_cpp").field("n", r.patterns.size()).field("patterns", r.patterns)
        .field("seconds", clock.seconds()).field("cliques", r.cliques).field("hash_keys", r.hashKeys)
        .field("evaluated", r.evaluated).field("all_pattern_num", r.generatedCandidates).field("ALL", r.normalizer)
        .field("max_rss_kb", peakMemoryKb());
    std::cout << out.str() << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "pso_mucp: " << e.what() << "\n";
        return kExitUsage;
    }
}
