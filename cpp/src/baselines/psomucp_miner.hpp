// PSO-MUCP (Bao et al., Data Min. Knowl. Disc. 40:38, 2026): high-utility co-location mining with one utility per
// feature, which may be negative. C++ port of the authors' HUCPM/PSO-MUCP.py (github.com/yuanshuaikang/PSO-MUCP).
// Each private method is one function of the Python code and keeps its iteration orders:
//   totalUtility (Compute_utility), buildNeighborhoods (grid_method), pruneFeatures (Prune_one),
//   enumerateCliques (Enum_Cliques + con_hash_table), mineLevels (Find_High_utility_pattern),
//   generateCandidates (gener_k).
// Python lists searched with `in` are hash sets here; this changes the cost of a membership test, not its result.
#pragma once

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/csv_table.hpp"

namespace colocation {

struct PsoInstance {
    char feature;      // first character of the Feature column
    std::string id;    // feature + instance number, as in the Python code
    double x, y;
};

struct PsoMucpConfig {
    std::map<char, double> utility;          // v(f) of every feature
    double minUtility = 0.0;                  // minutil
    double distance = 0.0;                    // D
    std::optional<double> normalizer;         // ALL; default: sum over features of count x v(f), as in the code
    static std::map<char, double> parseUtility(std::string text);   // "A:2,B:-1" or "{'A': 2, 'B': -1}"
};

struct PsoMucpResult {
    std::vector<std::string> patterns;        // sorted characters, sorted list, size >= 2
    long long cliques = 0;                    // cliques enumerated (before merging into the hash table)
    long long hashKeys = 0;
    long long evaluated = 0;                  // candidates whose utility was computed, all levels
    long long generatedCandidates = 0;        // all_pattern_num of the Python code (levels >= 2)
    double normalizer = 0.0;
};

class PsoMucpMiner {
public:
    PsoMucpMiner(std::vector<PsoInstance> instances, PsoMucpConfig config);
    static std::vector<PsoInstance> loadInstances(const CsvTable& table);
    PsoMucpResult mine();

private:
    struct HashEntry {                        // inner dict of one hash key: feature -> instances, insertion order
        std::vector<char> features;
        std::map<char, std::vector<int>> instances;
        std::map<char, std::unordered_set<int>> seen;
    };
    struct Candidate {
        double upperBound = 0;                // PUUR numerator
        double utility = 0;                   // lambda numerator
    };

    std::vector<PsoInstance> instances_;
    PsoMucpConfig config_;
    double normalizer_ = 0;
    // neighbourhoods (grid_method) and pruning (Prune_one)
    std::vector<int> neighborhoodOrder_;
    std::vector<std::vector<int>> neighbors_;
    std::vector<char> positive_, negative_, deleted_, sortList_;
    std::vector<int> keptOrder_;
    std::vector<std::vector<int>> keptNeighbors_;
    std::vector<std::unordered_set<int>> keptNeighborSet_;
    // clique hash table (con_hash_table)
    std::vector<std::string> hashKeys_;
    std::unordered_map<std::string, int> hashIndex_;
    std::vector<HashEntry> table_;
    long long cliqueCount_ = 0;

    double utilityOf(char feature) const;
    double totalUtility() const;
    void buildNeighborhoods();
    void pruneFeatures();
    void enumerateCliques();
    void addClique(int key, const std::vector<int>& clique);
    std::map<std::string, Candidate> scoreCandidates(const std::vector<std::string>& candidates) const;
    std::vector<std::string> generateCandidates(const std::vector<std::string>& candidates,
                                                const std::map<std::string, Candidate>& scores,
                                                std::vector<std::string>& noHope) const;
    bool laterInOrder(char feature, const std::string& pattern) const;   // order(sort_list, elem_1, elem_2)
    bool keyContains(const std::string& key, const std::string& pattern) const;
    static bool contains(const std::vector<char>& v, char c);
};

}  // namespace colocation
