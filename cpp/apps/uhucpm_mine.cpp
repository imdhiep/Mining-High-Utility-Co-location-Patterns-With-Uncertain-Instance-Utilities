// One UHUCPM run: reads an instance file (Feature,Instance,LocX,LocY,...) and a posterior file, mines all patterns with
// Lambda_z >= gamma and prints one JSON line.
// Usage: uhucpm_mine DATA_CSV POSTERIOR GAMMA D Z [--bound two|exact|mean|none] [--rule global|sum]
//                    [--deadline SEC] [--scores] [--top K] [--set-level DELTA]
//   --top K           the patterns ranked by Lambda_z down to the K-th (ties included), searching from threshold GAMMA
//   --set-level DELTA the set rule S_delta: candidates are the patterns whose point score (z = 0) reaches GAMMA; the
//                     output also lists the probability p of every selected pattern
//   --top-probability K  the patterns ranked by p = P(Lambda >= GAMMA) down to the K-th (ties included); the scores
//                     printed are the probabilities p (Z argument ignored)
// Exit code 3 when the deadline was reached (the pattern list is then incomplete).
#include <algorithm>
#include <iostream>
#include <map>
#include <stdexcept>

#include "common/command_line.hpp"
#include "common/csv_table.hpp"
#include "common/json_writer.hpp"
#include "common/process_stats.hpp"
#include "common/utility_posterior.hpp"
#include "mining/set_rule.hpp"
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
    MiningResult result;
    std::vector<double> probabilities;   // set rule only
    if (cl.has("set-level")) {
        MinerConfig point = config;
        point.z = 0.0;
        const MiningResult pool = UhucpmMiner(table, utilities, point).mine();
        std::vector<std::vector<int>> patterns;
        std::vector<double> M, V;
        for (const auto& kv : pool.patterns) {
            const std::vector<int> instances = table.participatingInstances(kv.first);
            patterns.push_back(kv.first);
            M.push_back(utilities.totalMean(instances));
            V.push_back(utilities.totalVariance(instances));
        }
        const SetSelection sel = SetRule::select(M, V, config.gamma * utilities.normalizer(), cl.option("set-level", 0.05));
        result.statistics = pool.statistics;
        std::map<std::vector<int>, double> p;
        for (std::size_t i = 0; i < patterns.size(); ++i)
            if (sel.selected[i]) {
                result.patterns[patterns[i]] = pool.patterns.at(patterns[i]);
                p[patterns[i]] = sel.probability[i];
            }
        for (const auto& kv : result.patterns) probabilities.push_back(p[kv.first]);
    } else if (cl.has("top-probability")) {
        result = UhucpmMiner(table, utilities, config).mineTopProbability(cl.option("top-probability", 0));
    } else if (cl.has("top")) {
        result = UhucpmMiner(table, utilities, config).mineTop(cl.option("top", 0), config.gamma);
    } else {
        result = UhucpmMiner(table, utilities, config).mine();
    }
    struct Named {
        std::string name;
        double score, probability;
        bool operator<(const Named& o) const { return name < o.name; }
    };
    std::vector<Named> named;
    std::size_t index = 0;
    for (const auto& kv : result.patterns) {
        named.push_back({data.patternName(kv.first), kv.second, probabilities.empty() ? 0.0 : probabilities[index]});
        ++index;
    }
    std::sort(named.begin(), named.end());
    std::vector<std::string> names;
    std::vector<double> scores, prob;
    for (const auto& p : named) {
        names.push_back(p.name);
        scores.push_back(p.score);
        prob.push_back(p.probability);
    }
    const MinerStatistics& st = result.statistics;
    JsonLineWriter out;
    out.field("mode", "uhucpm_cpp_" + boundName).field("n", names.size()).field("patterns", names);
    if (cl.has("scores")) out.field("scores", scores);
    if (cl.has("set-level")) out.field("probabilities", prob);
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
