// UHUCPM (Algorithms 1 and 2 of the paper): all patterns P with Lambda_z(P) >= gamma, exactly.
//   1. feature bounds over the cliques of each feature, features below gamma removed until none is (steps 3-6);
//   2. features ordered by decreasing feature bound, features whose instances all have mu <= 0 last (steps 7-8);
//   3. depth-first search over the set-enumeration tree on the clique table; at each node the suffix candidate sets are
//      built in one pass and the first child whose suffix bound is below gamma Z is found by binary search.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "mining/instance_utilities.hpp"
#include "mining/subtree_bound.hpp"
#include "spatial/clique_table.hpp"

namespace colocation {

// the unsafe local pruning rule of Section III-D, available only to measure how many patterns it would miss
enum class LocalRule { None, AllNonPositive, SummedMeanNonPositive };
LocalRule parseLocalRule(const std::string& name);   // "", "global", "sum"

struct MinerConfig {
    double gamma = 0.0;                 // threshold on the normalized score
    double z = 1.645;                   // z of the lower bound
    BoundTest boundTest = BoundTest::MeanChordPrefix;
    LocalRule localRule = LocalRule::None;
    double deadlineSeconds = 0.0;       // 0: no deadline
};

struct MinerStatistics {
    long long visited = 0;              // nodes of the search tree
    long long evaluated = 0;            // patterns whose score was computed
    long long boundCalls = 0;
    long long probes = 0;               // bound evaluations of the binary search
    long long subtreePruned = 0;        // nodes whose whole subtree was pruned
    long long suffixPruned = 0;         // nodes where some children were pruned
    long long nonPositiveStops = 0;     // nodes where every remaining feature has only non-positive instances
    long long localRuleSkips = 0;
    bool timedOut = false;
};

struct MiningResult {
    std::map<std::vector<int>, double> patterns;   // sorted feature indices -> score
    MinerStatistics statistics;
};

class UhucpmMiner {
public:
    // scores within kThresholdTolerance below gamma count as reaching it
    static constexpr double kThresholdTolerance = 1e-12;

    UhucpmMiner(const CliqueTable& table, const InstanceUtilities& utilities, MinerConfig config);
    // all patterns with Lambda_z >= config.gamma
    MiningResult mine();
    // the patterns ranked by Lambda_z down to the k-th: mines at startGamma, halves the threshold until at least k
    // patterns reach it (or it falls below kSmallestTopGamma), and keeps the patterns whose score is at least the k-th
    // largest score (ties included; all patterns found when fewer than k)
    MiningResult mineTop(int k, double startGamma);
    static constexpr double kSmallestTopGamma = 1e-12;

private:
    class Search;                       // state of one run of the depth-first search
    const CliqueTable& table_;
    const InstanceUtilities& utilities_;
    MinerConfig config_;
    SubtreeBound bound_;
};

}  // namespace colocation
