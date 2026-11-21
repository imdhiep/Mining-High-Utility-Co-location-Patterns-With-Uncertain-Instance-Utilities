#include <random>

#include "mining/set_rule.hpp"
#include "mining/uhucpm_miner.hpp"
#include "posterior/gamma_poisson_model.hpp"
#include "posterior/npmle.hpp"
#include "spatial/clique_table.hpp"
#include "stats/special_functions.hpp"
#include "test_data.hpp"
#include "test_framework.hpp"

using namespace colocation;

TEST(special_functions_reference_values) {
    // reference values: digamma(1) = -Euler gamma, digamma(1/2) = -gamma - 2 ln 2, trigamma(1) = pi^2 / 6, trigamma(10)
    CHECK_NEAR(SpecialFunctions::digamma(1.0), -0.5772156649015329, 1e-13);
    CHECK_NEAR(SpecialFunctions::digamma(0.5), -1.9635100260214235, 1e-13);
    CHECK_NEAR(SpecialFunctions::digamma(1e6), std::log(1e6) - 0.5e-6, 1e-12);
    CHECK_NEAR(SpecialFunctions::trigamma(1.0), 1.6449340668482264, 1e-12);
    CHECK_NEAR(SpecialFunctions::trigamma(10.0), 0.10516633568168575, 1e-13);
    CHECK_NEAR(SpecialFunctions::normalCdf(1.645), 0.95001509446087862, 1e-12);
    CHECK_THROWS(SpecialFunctions::digamma(0.0));
}

TEST(set_rule_keeps_the_largest_admissible_prefix) {
    std::mt19937_64 rng(21);
    std::normal_distribution<double> n01;
    std::uniform_real_distribution<double> var(0.05, 2.0);
    for (int it = 0; it < 500; ++it) {
        const int n = 1 + it % 25;
        std::vector<double> M, V;
        for (int i = 0; i < n; ++i) {
            M.push_back(1.0 + n01(rng));
            V.push_back(var(rng));
        }
        const double delta = (it % 4 + 1) * 0.05;
        const SetSelection s = SetRule::select(M, V, 1.0, delta);
        std::vector<double> p(n);
        for (int i = 0; i < n; ++i) p[i] = SpecialFunctions::normalCdf((M[i] - 1.0) / std::sqrt(V[i]));
        std::vector<double> sorted = p;
        std::sort(sorted.begin(), sorted.end(), std::greater<double>());
        int best = 0;
        double missed = 0;
        for (int k = 1; k <= n; ++k) {
            missed += 1 - sorted[k - 1];
            if (missed / k <= delta) best = k;
        }
        int count = 0;
        double minSelected = 2, maxOther = -1;
        for (int i = 0; i < n; ++i) {
            CHECK_NEAR(s.probability[i], p[i], 1e-15);
            if (s.selected[i]) {
                ++count;
                minSelected = std::min(minSelected, p[i]);
            } else {
                maxOther = std::max(maxOther, p[i]);
            }
        }
        CHECK(count == best);
        CHECK(count == 0 || count == n || minSelected >= maxOther);
    }
    CHECK(SetRule::select({}, {}, 0.0, 0.05).selected.empty());
}

TEST(mine_top_equals_brute_force_ranking) {
    std::mt19937_64 rng(22);
    for (int it = 0; it < 150; ++it) {
        const auto data = unit::randomDataset(rng, 15 + it % 40, 3 + it % 5, 60.0);
        const double d = 14.0 + (it % 3) * 6.0;
        const UtilityPosterior p = unit::randomPosterior(rng, data.size(), it % 2 ? 0 : 8, 0.5);
        if (unit::positiveSum(p.mean) <= 0) continue;
        const CliqueTable table = CliqueTable::build(data, d);
        const InstanceUtilities u(p);
        const double normalizer = unit::positiveSum(p.mean);
        std::vector<double> positive;
        for (const auto& kv : unit::brutePatterns(data, d)) {
            const double s = unit::bruteScore(p, kv.second, 1.645, normalizer);
            if (s >= UhucpmMiner::kSmallestTopGamma) positive.push_back(s);
        }
        std::sort(positive.begin(), positive.end(), std::greater<double>());
        for (int k : {1, 5, 20}) {
            MinerConfig c;
            c.z = 1.645;
            const MiningResult r = UhucpmMiner(table, u, c).mineTop(k, 0.05);
            if (static_cast<int>(positive.size()) >= k) {
                const double kth = positive[k - 1];
                std::size_t expected = 0;
                for (double s : positive) expected += s >= kth;
                CHECK(r.patterns.size() == expected);
                for (const auto& kv : r.patterns) CHECK(kv.second >= kth - 1e-12);
            } else {
                // fewer than k patterns above the smallest threshold: every pattern found at the last threshold
                for (const auto& kv : r.patterns) CHECK(kv.second >= 0.0);
            }
        }
    }
}

TEST(prior_median_of_a_grid_prior) {
    const std::vector<double> grid = {0, 1, 2, 3};
    CHECK(RateCenter::priorMedian(grid, {0.1, 0.3, 0.4, 0.2}) == 2);
    CHECK(RateCenter::priorMedian(grid, {0.25, 0.25, 0.25, 0.25}) == 1);
    CHECK(parseCenterRule("median") == CenterRule::PriorMedian);
    CHECK_THROWS(parseCenterRule("mode"));
}

TEST(gamma_poisson_model_matches_closed_form_and_monte_carlo) {
    std::mt19937_64 rng(23);
    std::gamma_distribution<double> rates(2.0, 3.0);
    std::vector<double> counts;
    std::vector<int> features;
    const double E = 2.0;
    for (int i = 0; i < 400; ++i) {
        std::poisson_distribution<int> pois(rates(rng) * E);
        counts.push_back(pois(rng));
        features.push_back(i % 2);
    }
    GammaPoissonPosteriorModel model(counts, features, E);
    const PosteriorEstimate e = model.estimate();
    for (int f = 0; f < 2; ++f) {
        std::vector<double> x;
        for (int i = f; i < 400; i += 2) x.push_back(counts[i]);
        double m = 0, v = 0;
        for (double c : x) m += c / x.size();
        for (double c : x) v += (c - m) * (c - m) / (x.size() - 1);
        const double A = v > m ? m * m / (v - m) : 1e6, beta = A * E / m, c = RateCenter::fromCounts(x, E);
        for (int a = 0; a < 3; ++a) {
            const int i = f + 2 * a;
            CHECK_NEAR(e.mean[i], SpecialFunctions::digamma(A + counts[i]) - std::log(beta + E) - c, 1e-12);
            CHECK_NEAR(e.variance[i], SpecialFunctions::trigamma(A + counts[i]), 1e-12);
            // E[(ln lambda - c)+] by sampling lambda ~ Gamma(A + x, beta + E)
            std::gamma_distribution<double> post(A + counts[i], 1.0 / (beta + E));
            double s = 0;
            const int draws = 400000;
            for (int k = 0; k < draws; ++k) s += std::max(std::log(post(rng)) - c, 0.0);
            CHECK_NEAR(e.expectedPositive[i], s / draws, 5e-3);
        }
    }
    const BootstrapMatrix b = model.bootstrap(5, 9);
    CHECK(b.samples() == 5 && b.instances() == 400);
    CHECK(model.bootstrap(5, 9).values() == b.values());
}

TEST(mine_top_probability_equals_brute_force_ranking) {
    std::mt19937_64 rng(24);
    int checked = 0;
    for (int it = 0; it < 120; ++it) {
        const auto data = unit::randomDataset(rng, 15 + it % 35, 3 + it % 4, 60.0);
        const double d = 14.0 + (it % 3) * 6.0;
        const UtilityPosterior p = unit::randomPosterior(rng, data.size(), it % 2 ? 0 : 8, 0.5);
        const double normalizer = unit::positiveSum(p.mean);
        if (normalizer <= 0) continue;
        const CliqueTable table = CliqueTable::build(data, d);
        const InstanceUtilities u(p);
        const double gamma = 0.03 + 0.02 * (it % 4);
        std::vector<double> probs;
        for (const auto& kv : unit::brutePatterns(data, d)) {
            const double m = unit::bruteScore(p, kv.second, 0.0, 1.0);                         // M
            const double v = std::pow((m - unit::bruteScore(p, kv.second, 1.0, 1.0)), 2.0);   // V
            const double pr = SpecialFunctions::normalCdf((m - gamma * normalizer) / std::sqrt(v));
            if (pr >= SpecialFunctions::normalCdf(UhucpmMiner::kTopZEnd)) probs.push_back(pr);
        }
        std::sort(probs.begin(), probs.end(), std::greater<double>());
        for (int k : {1, 5, 15}) {
            if (static_cast<int>(probs.size()) < k) continue;
            MinerConfig c;
            c.gamma = gamma;
            const MiningResult r = UhucpmMiner(table, u, c).mineTopProbability(k);
            const double kth = probs[k - 1];
            std::size_t expected = 0;
            for (double x : probs) expected += x >= kth - 1e-12;
            CHECK(r.patterns.size() == expected);
            for (const auto& kv : r.patterns) CHECK(kv.second >= kth - 1e-12);
            ++checked;
        }
    }
    CHECK(checked > 100);
}
