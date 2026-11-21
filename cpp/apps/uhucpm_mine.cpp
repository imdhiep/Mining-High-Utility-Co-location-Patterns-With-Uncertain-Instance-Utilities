// One UHUCPM run: reads an instance file (Feature,Instance,LocX,LocY,...) and a posterior file, mines all patterns with
// Lambda_z >= gamma and prints one JSON line.
// Usage: uhucpm_mine DATA_CSV POSTERIOR GAMMA D Z [--bound two|exact|mean|none] [--rule global|sum]
//                    [--deadline SEC] [--scores]
// Exit code 3 when the deadline was reached (the pattern list is then incomplete).
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "common/command_line.hpp"
#include "common/csv_table.hpp"
#include "common/json_writer.hpp"
#include "common/process_stats.hpp"
#include "common/utility_posterior.hpp"
#include "mining/uhucpm_miner.hpp"
#include "spatial/clique_table.hpp"
#include "spatial/spatial_dataset.hpp"

using namespace colocation;

namespace {

constexpr int kExitUsage = 2;
constexpr int kExitTimeout = 3;

int run(int argc, char** argv) {
    const Stopwatch clock;
    const CommandLine cl(argc, argv, {"scores"});
    if (cl.positionalCount() < 5) {
        std::cerr << "usage: uhucpm_mine DATA_CSV POSTERIOR GAMMA D Z [--bound B] [--rule R] [--deadline S] [--scores]\n";
        return kExitUsage;
    }
    MinerConfig config;
    config.gamma = cl.positionalDouble(2);
    const double distance = cl.positionalDouble(3);
    config.z = cl.positionalDouble(4);
    const std::string boundName = cl.option("bound", std::string("two"));
    config.boundTest = parseBoundTest(boundName);
    config.localRule = parseLocalRule(cl.option("rule", std::string("")));
    config.deadlineSeconds = cl.option("deadline", 0.0);

    const SpatialDataset data = SpatialDataset::fromTable(CsvTable::read(cl.positional(0)));
    const UtilityPosterior posterior = UtilityPosteriorFile::read(cl.positional(1));
    if (static_cast<int>(posterior.size()) != data.size())
        throw std::runtime_error("posterior size " + std::to_string(posterior.size()) + " != " + std::to_string(data.size()));
    const CliqueTable table = CliqueTable::build(data, distance);
    const InstanceUtilities utilities(posterior);
    const MiningResult result = UhucpmMiner(table, utilities, config).mine();

    std::vector<std::pair<std::string, double>> named;
    for (const auto& kv : result.patterns) named.emplace_back(data.patternName(kv.first), kv.second);
    std::sort(named.begin(), named.end());
    std::vector<std::string> names;
    std::vector<double> scores;
    for (const auto& p : named) {
        names.push_back(p.first);
        scores.push_back(p.second);
    }
    const MinerStatistics& st = result.statistics;
    JsonLineWriter out;
    out.field("mode", "uhucpm_cpp_" + boundName).field("n", names.size()).field("patterns", names);
    if (cl.has("scores")) out.field("scores", scores);
    out.field("seconds", clock.seconds()).field("cliques", table.rows()).field("evaluated", st.evaluated)
        .field("visited", st.visited).field("Z", utilities.normalizer()).field("timeout", st.timedOut)
        .field("max_rss_kb", peakMemoryKb());
    std::cout << out.str() << std::endl;
    return st.timedOut ? kExitTimeout : 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "uhucpm_mine: " << e.what() << "\n";
        return kExitUsage;
    }
}
