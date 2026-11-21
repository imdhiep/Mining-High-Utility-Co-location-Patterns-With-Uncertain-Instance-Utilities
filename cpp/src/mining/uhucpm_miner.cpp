#include "mining/uhucpm_miner.hpp"

#include <algorithm>
#include <stdexcept>

#include "common/process_stats.hpp"

namespace colocation {

LocalRule parseLocalRule(const std::string& name) {
    if (name.empty()) return LocalRule::None;
    if (name == "global") return LocalRule::AllNonPositive;
    if (name == "sum") return LocalRule::SummedMeanNonPositive;
    throw std::invalid_argument("unknown local rule " + name);
}

namespace {

struct SearchTimeout {};

}  // namespace

// One depth-first search. Scratch arrays are allocated once and reused at every node.
class UhucpmMiner::Search {
public:
    Search(const UhucpmMiner& miner, MiningResult& result)
        : table_(miner.table_), u_(miner.utilities_), cfg_(miner.config_), bound_(miner.bound_), result_(result),
          stats_(result.statistics), k_(table_.features()), stamp_(u_.size(), 0),
          bestKey_(std::max(u_.positiveCount(), 1), -1), alive_(k_, 0), nonPositive_(k_, 0), featureBound_(k_, 0.0),
          position_(k_, -1) {}

    void run() {
        for (int f = 0; f < k_; ++f) alive_[f] = !table_.rowsWithFeature(f).empty();
        removeFeaturesBelowGamma();
        markNonPositiveFeatures();
        orderFeatures();
        try {
            for (int f : order_) {
                std::vector<int> pattern{f};
                visit(pattern, table_.rowsWithFeature(f));
            }
        } catch (const SearchTimeout&) {
            stats_.timedOut = true;
        }
    }

private:
    const CliqueTable& table_;
    const InstanceUtilities& u_;
    const MinerConfig& cfg_;
    const SubtreeBound& bound_;
    MiningResult& result_;
    MinerStatistics& stats_;
    const int k_;
    Stopwatch clock_;
    std::vector<int> stamp_;           // per-instance marker for deduplication
    int stampId_ = 0;
    std::vector<int> bestKey_;         // per rank: largest suffix position at which the instance is a candidate
    std::vector<char> alive_, nonPositive_;
    std::vector<double> featureBound_;
    std::vector<int> order_, position_;

    double threshold() const { return cfg_.gamma - UhucpmMiner::kThresholdTolerance; }

    // instances of the given features in the given rows, each once
    std::vector<int> collectInstances(const std::vector<int>& rows, const std::vector<int>& features) {
        ++stampId_;
        std::vector<int> out;
        for (int c : rows)
            for (int f : features) {
                const int a = table_.at(c, f);
                if (a != CliqueTable::kNone && stamp_[a] != stampId_) {
                    stamp_[a] = stampId_;
                    out.push_back(a);
                }
            }
        return out;
    }

    // bound of a set of instances, through their ranks
    double boundOfInstances(const std::vector<int>& instances) const {
        std::vector<int> ranks;
        for (int a : instances)
            if (u_.rank(a) >= 0) ranks.push_back(u_.rank(a));
        std::sort(ranks.begin(), ranks.end());
        std::vector<double> m, v;
        for (int r : ranks) {
            m.push_back(u_.rankedMean(r));
            v.push_back(u_.rankedVariance(r));
        }
        return bound_.value(m, v);
    }

    // steps 3-6 of Algorithm 1: feature bounds over the cliques of each feature, restricted to the alive features
    void removeFeaturesBelowGamma() {
        std::vector<int> aliveList;
        while (true) {
            aliveList.clear();
            for (int f = 0; f < k_; ++f)
                if (alive_[f]) aliveList.push_back(f);
            for (int f : aliveList) {
                featureBound_[f] = boundOfInstances(collectInstances(table_.rowsWithFeature(f), aliveList)) / u_.normalizer();
                ++stats_.boundCalls;
            }
            if (!bound_.prunes()) return;
            bool removed = false;
            for (int f : aliveList)
                if (featureBound_[f] < threshold()) {
                    alive_[f] = 0;
                    removed = true;
                }
            if (!removed) return;
        }
    }

    // step 7: features whose instances (in cliques) all have a non-positive mean
    void markNonPositiveFeatures() {
        for (int f = 0; f < k_; ++f) {
            if (!alive_[f]) continue;
            bool allNonPositive = true;
            for (int c : table_.rowsWithFeature(f))
                if (u_.mean(table_.at(c, f)) > 0) {
                    allNonPositive = false;
                    break;
                }
            nonPositive_[f] = allNonPositive;
        }
    }

    // step 8: decreasing feature bound, ties by index, non-positive features last
    void orderFeatures() {
        for (int tier = 0; tier < 2; ++tier) {
            std::vector<int> group;
            for (int f = 0; f < k_; ++f)
                if (alive_[f] && static_cast<bool>(nonPositive_[f]) == (tier == 1)) group.push_back(f);
            std::sort(group.begin(), group.end(), [&](int a, int b) {
                return featureBound_[a] != featureBound_[b] ? featureBound_[a] > featureBound_[b] : a < b;
            });
            order_.insert(order_.end(), group.begin(), group.end());
        }
        for (std::size_t i = 0; i < order_.size(); ++i) position_[order_[i]] = static_cast<int>(i);
    }

    void checkDeadline() {
        if (cfg_.deadlineSeconds > 0 && stats_.visited % 64 == 0 && clock_.seconds() > cfg_.deadlineSeconds) throw SearchTimeout{};
    }

    bool allNonPositive(const std::vector<int>& pattern, const std::vector<int>& tail) const {
        for (int f : pattern)
            if (!nonPositive_[f]) return false;
        for (int f : tail)
            if (!nonPositive_[f]) return false;
        return true;
    }

    // Suffix candidate sets of a node in one pass (Section V-B). Every candidate instance gets the largest position j
    // of T for which it lies in I_{T[j]}(P); the instances come out of the rank-indexed marker array in rank order.
    // Returns true when the bound of the whole subtree prunes; otherwise firstPruned is the first child position whose
    // suffix bound is below the threshold (|T| if none), found by binary search (the bounds do not increase in j).
    bool suffixSearch(const std::vector<int>& rows, const std::vector<int>& pattern, const std::vector<int>& tail,
                      double thresholdTimesZ, int& firstPruned) {
        const int nT = static_cast<int>(tail.size());
        int lowRank = u_.positiveCount(), highRank = -1;
        auto mark = [&](int instance, int key) {
            const int r = u_.rank(instance);
            if (r < 0 || key < 0) return;
            bestKey_[r] = std::max(bestKey_[r], key);
            lowRank = std::min(lowRank, r);
            highRank = std::max(highRank, r);
        };
        for (int c : rows) {
            int lastTail = -1;
            for (int j = 0; j < nT; ++j)
                if (table_.has(c, tail[j])) lastTail = j;
            for (int f : pattern) mark(table_.at(c, f), lastTail);
            for (int j = 0; j < nT; ++j)
                if (table_.has(c, tail[j])) mark(table_.at(c, tail[j]), j);
        }
        std::vector<double> means, variances;
        std::vector<int> keys;
        for (int r = lowRank; r <= highRank; ++r)
            if (bestKey_[r] >= 0) {
                means.push_back(u_.rankedMean(r));
                variances.push_back(u_.rankedVariance(r));
                keys.push_back(bestKey_[r]);
                bestKey_[r] = -1;
            }
        if (means.empty()) {
            const bool prunes = 0.0 < thresholdTimesZ;
            firstPruned = prunes ? 0 : nT;
            return prunes;
        }
        std::vector<double> m, v;
        auto belowAt = [&](int j) {
            m.clear();
            v.clear();
            for (std::size_t i = 0; i < means.size(); ++i)
                if (keys[i] >= j) {
                    m.push_back(means[i]);
                    v.push_back(variances[i]);
                }
            return bound_.below(m, v, thresholdTimesZ);
        };
        ++stats_.probes;
        if (belowAt(0)) {
            firstPruned = 0;
            return true;
        }
        int lo = 1, hi = nT;   // belowAt(lo - 1) is false and the answer lies in [lo, hi]
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            ++stats_.probes;
            if (belowAt(mid)) hi = mid;
            else lo = mid + 1;
        }
        firstPruned = lo;
        return false;
    }

    // the unsafe local rule of Section III-D (audit only)
    bool skippedByLocalRule(int child, const std::vector<int>& childRows, bool scored, double score) {
        if (cfg_.localRule == LocalRule::None || !scored || score >= threshold()) return false;
        if (cfg_.localRule == LocalRule::AllNonPositive) return nonPositive_[child];
        double summed = 0;
        for (int a : collectInstances(childRows, {child})) summed += u_.mean(a);
        return summed <= 0;
    }

    void visit(std::vector<int>& pattern, const std::vector<int>& rows) {
        ++stats_.visited;
        checkDeadline();
        bool scored = false;
        double score = 0;
        if (pattern.size() >= 2) {
            score = u_.score(collectInstances(rows, pattern), cfg_.z);
            scored = true;
            ++stats_.evaluated;
            if (score >= threshold()) {
                std::vector<int> key = pattern;
                std::sort(key.begin(), key.end());
                result_.patterns[key] = score;
            }
        }
        const std::vector<int> tail(order_.begin() + position_[pattern.back()] + 1, order_.end());
        if (tail.empty()) return;
        if (bound_.prunes() && cfg_.gamma > UhucpmMiner::kThresholdTolerance && allNonPositive(pattern, tail)) {
            ++stats_.nonPositiveStops;
            return;
        }
        std::vector<int> tailRows;
        for (int c : rows)
            for (int g : tail)
                if (table_.has(c, g)) {
                    tailRows.push_back(c);
                    break;
                }
        if (tailRows.empty()) return;
        int stop = static_cast<int>(tail.size());
        if (bound_.prunes()) {
            int firstPruned = 0;
            const bool subtreePruned = suffixSearch(tailRows, pattern, tail, threshold() * u_.normalizer(), firstPruned);
            ++stats_.boundCalls;
            if (subtreePruned) {
                ++stats_.subtreePruned;
                return;
            }
            if (firstPruned < stop) {
                stop = firstPruned;
                ++stats_.suffixPruned;
            }
        }
        for (int i = 0; i < stop; ++i) {
            const int child = tail[i];
            std::vector<int> childRows;
            for (int c : tailRows)
                if (table_.has(c, child)) childRows.push_back(c);
            if (childRows.empty()) continue;
            if (skippedByLocalRule(child, childRows, scored, score)) {
                ++stats_.localRuleSkips;
                continue;
            }
            pattern.push_back(child);
            visit(pattern, childRows);
            pattern.pop_back();
        }
    }
};

UhucpmMiner::UhucpmMiner(const CliqueTable& table, const InstanceUtilities& utilities, MinerConfig config)
    : table_(table), utilities_(utilities), config_(config), bound_(config.boundTest, config.z) {
    if (table.instanceCount() != utilities.size()) throw std::invalid_argument("clique table and utilities differ in size");
}

MiningResult UhucpmMiner::mine() {
    MiningResult result;
    Search(*this, result).run();
    return result;
}

}  // namespace colocation
