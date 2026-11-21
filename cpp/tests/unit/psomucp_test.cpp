#include <random>

#include "baselines/psomucp_miner.hpp"
#include "test_data.hpp"
#include "test_framework.hpp"

using namespace colocation;

namespace {

std::vector<PsoInstance> toPso(const SpatialDataset& d) {
    std::vector<PsoInstance> out;
    std::map<int, int> counter;
    for (int i = 0; i < d.size(); ++i) {
        const auto& o = d.instance(i);
        const char f = d.featureName(o.feature)[0];
        out.push_back({f, std::string(1, f) + std::to_string(++counter[o.feature]), o.x, o.y});
    }
    return out;
}

// lambda(P) = sum over f in P of v(f) |PI(P, f)| / ALL, ALL = sum over features of count x v(f)
std::vector<std::string> bruteFeatureUtility(const SpatialDataset& d, const std::map<char, double>& v, double minUtility,
                                             double dist) {
    double all = 0;
    for (int i = 0; i < d.size(); ++i) all += v.at(d.featureName(d.instance(i).feature)[0]);
    std::vector<std::string> out;
    for (const auto& kv : unit::brutePatterns(d, dist)) {
        double u = 0;
        for (int i : kv.second) u += v.at(d.featureName(d.instance(i).feature)[0]);
        if (u / all >= minUtility) {
            std::string name;
            for (int f : kv.first) name += d.featureName(f);
            out.push_back(name);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

TEST(psomucp_parses_both_utility_formats) {
    const auto a = PsoMucpConfig::parseUtility("A:2,B:-1.5");
    const auto b = PsoMucpConfig::parseUtility("{'A': 2, 'B': -1.5}");
    CHECK(a == b);
    CHECK(a.at('B') == -1.5);
}

// with positive utilities every pruning strategy of PSO-MUCP is safe, so the port must return exactly the patterns whose
// feature-level utility ratio reaches the threshold. Utilities are real numbers so that no ratio ties the threshold:
// the authors' code removes a feature when Up / ALL <= minutil but accepts a pattern when lambda >= minutil, so a
// pattern whose utility equals the bound of one of its features and ties the threshold is dropped.
TEST(psomucp_equals_brute_force_with_positive_utilities) {
    std::mt19937_64 rng(12);
    std::uniform_real_distribution<double> util(0.5, 9.5);
    int nonEmpty = 0;
    for (int it = 0; it < 300; ++it) {
        const int k = 2 + it % 6;
        const auto data = unit::randomDataset(rng, 8 + it % 45, k, 60.0);
        std::map<char, double> v;
        for (int j = 0; j < k; ++j) v['A' + j] = util(rng);
        for (double minUtility : {0.0, 0.05, 0.15, 0.3}) {
            const double dist = 10.0 + (it % 4) * 6.0;
            PsoMucpConfig c;
            c.utility = v;
            c.minUtility = minUtility;
            c.distance = dist;
            const auto expected = bruteFeatureUtility(data, v, minUtility, dist);
            nonEmpty += !expected.empty();
            CHECK(PsoMucpMiner(toPso(data), c).mine().patterns == expected);
        }
    }
    CHECK(nonEmpty > 300);
}

// with negative utilities Strategy 4 of PSO-MUCP is not safe; the port keeps the authors' logic, so it may miss
// patterns but never reports one below the threshold
TEST(psomucp_with_negative_utilities_never_reports_a_pattern_below_threshold) {
    std::mt19937_64 rng(13);
    std::uniform_int_distribution<int> util(-5, 9);
    for (int it = 0; it < 300; ++it) {
        const int k = 2 + it % 6;
        const auto data = unit::randomDataset(rng, 8 + it % 45, k, 60.0);
        std::map<char, double> v;
        double all = 0;
        for (int j = 0; j < k; ++j) v['A' + j] = util(rng);
        for (int i = 0; i < data.size(); ++i) all += v.at(data.featureName(data.instance(i).feature)[0]);
        if (all <= 0) continue;
        PsoMucpConfig c;
        c.utility = v;
        c.minUtility = 0.1;
        c.distance = 20.0;
        const auto expected = bruteFeatureUtility(data, v, c.minUtility, c.distance);
        for (const auto& p : PsoMucpMiner(toPso(data), c).mine().patterns)
            CHECK(std::binary_search(expected.begin(), expected.end(), p));
    }
}
