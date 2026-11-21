#include <random>

#include "mining/instance_utilities.hpp"
#include "mining/subtree_bound.hpp"
#include "mining/uhucpm_miner.hpp"
#include "spatial/clique_table.hpp"
#include "test_data.hpp"
#include "test_framework.hpp"

using namespace colocation;

namespace {

// max over all subsets of (sum mu - z sqrt(sum s2)), floored at 0 (empty set)
double bestSubset(const std::vector<double>& mu, const std::vector<double>& s2, double z) {
    const int n = static_cast<int>(mu.size());
    double best = 0;
    for (int mask = 1; mask < (1 << n); ++mask) {
        double m = 0, v = 0;
        for (int i = 0; i < n; ++i)
            if (mask >> i & 1) {
                m += mu[i];
                v += s2[i];
            }
        best = std::max(best, m - z * std::sqrt(v));
    }
    return best;
}

// patterns of the brute-force enumeration whose score reaches gamma
std::map<std::vector<int>, double> bruteMine(const SpatialDataset& data, const UtilityPosterior& p, double d, double gamma,
                                             double z) {
    const double normalizer = p.normalizer > 0 ? p.normalizer : unit::positiveSum(p.mean);
    std::map<std::vector<int>, double> out;
    for (const auto& kv : unit::brutePatterns(data, d)) {
        const double s = unit::bruteScore(p, kv.second, z, normalizer);
        if (s >= gamma - UhucpmMiner::kThresholdTolerance) out[kv.first] = s;
    }
    return out;
}

bool samePatterns(const std::map<std::vector<int>, double>& a, const std::map<std::vector<int>, double>& b, double tol) {
    if (a.size() != b.size()) return false;
    for (auto ia = a.begin(), ib = b.begin(); ia != a.end(); ++ia, ++ib)
        if (ia->first != ib->first || std::fabs(ia->second - ib->second) > tol) return false;
    return true;
}

}  // namespace

TEST(prefix_scan_is_the_best_subset_for_rank_ordered_positive_means) {
    std::mt19937_64 rng(4);
    std::uniform_real_distribution<double> mu(0.01, 2.0), s2(0.0, 1.5);
    for (int it = 0; it < 2000; ++it) {
        const int n = 1 + it % 12;
        std::vector<std::pair<double, std::pair<double, double>>> items;
        for (int i = 0; i < n; ++i) {
            double m = mu(rng), v = it % 10 == 0 ? 0.25 : s2(rng);   // some ties in sigma^2
            items.push_back({v / m, {m, v}});
        }
        std::sort(items.begin(), items.end());
        std::vector<double> means, vars;
        for (const auto& x : items) {
            means.push_back(x.second.first);
            vars.push_back(x.second.second);
        }
        for (double z : {0.0, 1.0, 1.645, 3.0}) {
            const double exact = bestSubset(means, vars, z);
            CHECK_NEAR(SubtreeBound::prefixScan(means, vars, z), exact, 1e-10);
            CHECK(SubtreeBound::chordBound(means, vars, z) >= exact - 1e-10);
        }
    }
}

TEST(bound_decisions_are_valid) {
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> mu(0.01, 2.0), s2(0.0, 1.5), thr(0.0, 4.0);
    for (int it = 0; it < 2000; ++it) {
        const int n = 1 + it % 10;
        std::vector<std::pair<double, std::pair<double, double>>> items;
        for (int i = 0; i < n; ++i) {
            double m = mu(rng), v = s2(rng);
            items.push_back({v / m, {m, v}});
        }
        std::sort(items.begin(), items.end());
        std::vector<double> means, vars;
        for (const auto& x : items) {
            means.push_back(x.second.first);
            vars.push_back(x.second.second);
        }
        const double z = 1.645, t = thr(rng), exact = bestSubset(means, vars, z);
        for (BoundTest b : {BoundTest::MeanChordPrefix, BoundTest::MeanPrefix, BoundTest::MeanOnly}) {
            const SubtreeBound bound(b, z);
            if (bound.below(means, vars, t)) CHECK(exact < t + 1e-12);   // never prunes a set that can reach t
            CHECK(bound.value(means, vars) >= exact - 1e-10);
        }
        CHECK(!SubtreeBound(BoundTest::None, z).prunes());
    }
}

TEST(parse_names) {
    CHECK(parseBoundTest("two") == BoundTest::MeanChordPrefix);
    CHECK(parseBoundTest("exact") == BoundTest::MeanPrefix);
    CHECK(parseBoundTest("mean") == BoundTest::MeanOnly);
    CHECK(parseBoundTest("none") == BoundTest::None);
    CHECK_THROWS(parseBoundTest("fast"));
    CHECK(parseLocalRule("") == LocalRule::None);
    CHECK(parseLocalRule("global") == LocalRule::AllNonPositive);
    CHECK(parseLocalRule("sum") == LocalRule::SummedMeanNonPositive);
}

TEST(instance_scores_equal_direct_formula) {
    std::mt19937_64 rng(6);
    for (int it = 0; it < 300; ++it) {
        const int n = 3 + it % 30;
        const UtilityPosterior p = unit::randomPosterior(rng, n, it % 3 == 0 ? 0 : 2 + it % 19, 0.3);
        if (unit::positiveSum(p.mean) <= 0) continue;   // no positive mean: the default normalizer is 0
        const InstanceUtilities u(p);
        std::set<int> a;
        for (int i = 0; i < n; i += 1 + it % 3) a.insert(i);
        const std::vector<int> av(a.begin(), a.end());
        for (double z : {0.0, 1.645})
            CHECK_NEAR(u.score(av, z), unit::bruteScore(p, a, z, unit::positiveSum(p.mean)), 1e-10);
    }
}

TEST(miner_equals_brute_force) {
    std::mt19937_64 rng(7);
    int nonEmpty = 0;
    for (int it = 0; it < 400; ++it) {
        const auto data = unit::randomDataset(rng, 8 + it % 50, 2 + it % 6, 60.0);
        const double d = 10.0 + (it % 4) * 6.0;
        UtilityPosterior p = unit::randomPosterior(rng, data.size(), it % 2 ? 0 : 5 + it % 16, it % 3 ? 0.4 : -0.2);
        if (it % 5 == 0) p.normalizer = 7.5;
        const CliqueTable table = CliqueTable::build(data, d);
        const InstanceUtilities u(p);
        for (double z : {0.0, 1.645, 3.0})
            for (double gamma : {0.0, 0.02, 0.08, 0.2}) {
                const auto expected = bruteMine(data, p, d, gamma, z);
                nonEmpty += !expected.empty();
                for (BoundTest b : {BoundTest::MeanChordPrefix, BoundTest::MeanPrefix, BoundTest::MeanOnly, BoundTest::None}) {
                    MinerConfig c;
                    c.gamma = gamma;
                    c.z = z;
                    c.boundTest = b;
                    const MiningResult r = UhucpmMiner(table, u, c).mine();
                    CHECK(samePatterns(r.patterns, expected, 1e-9));
                }
            }
    }
    CHECK(nonEmpty > 1000);   // the cases are not trivially empty
}

TEST(miner_edge_cases) {
    std::mt19937_64 rng(8);
    const auto data = unit::randomDataset(rng, 40, 4, 50.0);
    const CliqueTable table = CliqueTable::build(data, 15.0);
    MinerConfig c;
    c.z = 1.645;
    // threshold above every score: empty result
    UtilityPosterior p = unit::randomPosterior(rng, data.size(), 10, 0.5);
    c.gamma = 1e6;
    CHECK(UhucpmMiner(table, InstanceUtilities(p), c).mine().patterns.empty());
    // every mean non-positive, positive threshold: empty result
    for (double& m : p.mean) m = -std::fabs(m) - 0.01;
    p.normalizer = 1.0;
    c.gamma = 0.01;
    CHECK(UhucpmMiner(table, InstanceUtilities(p), c).mine().patterns.empty());
    // distance with no neighbours: no clique, no pattern
    const CliqueTable empty = CliqueTable::build(data, 1e-6);
    c.gamma = 0.0;
    CHECK(empty.rows() == 0);
    CHECK(UhucpmMiner(empty, InstanceUtilities(p), c).mine().patterns.empty());
    // one feature: no pattern of size 2
    std::vector<SpatialInstance> inst = {{0, 0, 0}, {0, 1, 1}};
    const SpatialDataset one(inst, {"A"});
    UtilityPosterior q;
    q.mean = {1.0, 2.0};
    q.variance = {0.1, 0.1};
    CHECK(UhucpmMiner(CliqueTable::build(one, 5.0), InstanceUtilities(q), c).mine().patterns.empty());
}

TEST(local_rule_can_miss_patterns_but_never_adds_any) {
    std::mt19937_64 rng(9);
    int missed = 0;
    for (int it = 0; it < 300; ++it) {
        const auto data = unit::randomDataset(rng, 10 + it % 40, 3 + it % 4, 50.0);
        const UtilityPosterior p = unit::randomPosterior(rng, data.size(), 0, -0.1);
        const CliqueTable table = CliqueTable::build(data, 16.0);
        const InstanceUtilities u(p);
        MinerConfig c;
        c.gamma = 0.02;
        c.z = 1.0;
        const auto exact = UhucpmMiner(table, u, c).mine().patterns;
        for (LocalRule rule : {LocalRule::AllNonPositive, LocalRule::SummedMeanNonPositive}) {
            c.localRule = rule;
            const auto local = UhucpmMiner(table, u, c).mine().patterns;
            for (const auto& kv : local) CHECK(exact.count(kv.first));
            missed += static_cast<int>(exact.size() - local.size());
        }
    }
    CHECK(missed > 0);   // the running example of Section III-D is not an isolated case
}
