#include <cmath>
#include <random>

#include "posterior/count_posterior_model.hpp"
#include "posterior/npmle.hpp"
#include "posterior/rating_posterior_model.hpp"
#include "test_framework.hpp"

using namespace colocation;

namespace {

double digamma(double x) {
    double r = 0;
    while (x < 10) {
        r -= 1 / x;
        x += 1;
    }
    const double f = 1 / (x * x);
    return r + std::log(x) - 0.5 / x - f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f / 132))));
}

double trigamma(double x) {
    double r = 0;
    while (x < 10) {
        r += 1 / (x * x);
        x += 1;
    }
    const double f = 1 / (x * x);
    return r + 1 / x + f / 2 + f / x * (1.0 / 6 - f * (1.0 / 30 - f * (1.0 / 42 - f / 30)));
}

// Gamma(a, b) prior of lambda as weights on a log-rate grid: density of theta = ln lambda is prop. to exp(a theta - b e^theta)
std::vector<double> gammaPriorOnGrid(const std::vector<double>& grid, double a, double b) {
    std::vector<double> w(grid.size());
    double s = 0;
    for (std::size_t j = 0; j < grid.size(); ++j) s += w[j] = std::exp(a * grid[j] - b * std::exp(grid[j]));
    for (double& x : w) x /= s;
    return w;
}

}  // namespace

TEST(digamma_and_trigamma_reference_values) {
    CHECK_NEAR(digamma(1.0), -0.5772156649015329, 1e-12);
    CHECK_NEAR(digamma(0.5), -1.9635100260214235, 1e-12);
    CHECK_NEAR(trigamma(1.0), 1.6449340668482264, 1e-10);
    CHECK_NEAR(trigamma(10.0), 0.10516633568168575, 1e-10);
}

// Gamma-Poisson conjugacy: with a Gamma(a, b) prior and x ~ Poisson(lambda E), lambda | x ~ Gamma(a + x, b + E), so
// E[ln lambda | x] = digamma(a + x) - ln(b + E) and Var[ln lambda | x] = trigamma(a + x). The prior is chosen so that
// the counts are plausible under it (prior mean of lambda E = 225), the setting the local grid is designed for.
TEST(grid_posterior_matches_gamma_poisson_closed_form) {
    const double a = 3.0, b = 0.02, E = 1.5;
    const std::vector<double> counts = {0, 1, 2, 5, 30, 400};
    struct GridCase { int points; bool expectLocal; double tolMean, tolVar; };
    // fine grid: every instance on the grid; coarse grid: the counts 30 and 400 take the local-grid path
    for (const GridCase g : {GridCase{4000, false, 1e-6, 1e-6}, GridCase{300, true, 1e-4, 1e-4}}) {
        const std::vector<double> grid = LogRateGrid::linspace(-8.0, 9.0, g.points);
        const std::vector<double> prior = gammaPriorOnGrid(grid, a, b);
        const PoissonLikelihood L(counts, grid, E, 0);
        NpmleSettings s;
        const PosteriorMoments m = GridPosterior::compute(counts, E, grid, prior, 0, L, false, s);
        CHECK(m.local.empty() != g.expectLocal);
        for (const LocalPosterior& lp : m.local) CHECK(lp.edgeMass < 1e-6);
        for (std::size_t i = 0; i < counts.size(); ++i) {
            CHECK_NEAR(m.mean[i], digamma(a + counts[i]) - std::log(b + E), g.tolMean);
            CHECK_NEAR(m.variance[i], trigamma(a + counts[i]), g.tolVar);
        }
    }
}

// a prior far from the count pulls the posterior out of the local window; edgeMass reports it
TEST(local_grid_edge_mass_flags_a_prior_far_from_the_count) {
    const std::vector<double> counts = {400};
    const std::vector<double> grid = LogRateGrid::linspace(-8.0, 7.0, 300);
    const std::vector<double> prior = gammaPriorOnGrid(grid, 3.0, 2.0);   // prior mean of lambda E = 2.25
    const PoissonLikelihood L(counts, grid, 1.5, 0);
    const PosteriorMoments m = GridPosterior::compute(counts, 1.5, grid, prior, 0, L, false, NpmleSettings{});
    CHECK(m.local.size() == 1);
    CHECK(m.local[0].edgeMass > 1e-3);
}

TEST(poisson_log_survival) {
    CHECK_NEAR(PoissonLikelihood::logSurvival(0, 2.0), 0.0, 1e-15);
    CHECK_NEAR(PoissonLikelihood::logSurvival(1, 2.0), std::log(1 - std::exp(-2.0)), 1e-14);
    CHECK_NEAR(PoissonLikelihood::logSurvival(2, 0.5), std::log(1 - std::exp(-0.5) * 1.5), 1e-14);
    CHECK(std::isfinite(PoissonLikelihood::logSurvival(1, 1e-12)));
}

// the moment center recovers the location of log-normal rates: lambda = exp(c0 + s N(0, 1)), x ~ Poisson(lambda E)
TEST(moment_center_recovers_lognormal_location) {
    std::mt19937_64 rng(11);
    std::normal_distribution<double> n01;
    for (double c0 : {-1.0, 0.5, 2.0})
        for (double sd : {0.3, 1.0}) {
            const double E = 3.0;
            std::vector<double> x;
            for (int i = 0; i < 400000; ++i) {
                std::poisson_distribution<long long> pois(std::exp(c0 + sd * n01(rng)) * E);
                x.push_back(static_cast<double>(pois(rng)));
            }
            CHECK_NEAR(RateCenter::fromCounts(x, E), c0, 0.02);
        }
}

TEST(count_model_bootstrap_is_reproducible_and_shaped) {
    std::mt19937_64 rng(1);
    std::poisson_distribution<int> pois(3.0);
    std::vector<double> counts;
    std::vector<int> features;
    for (int i = 0; i < 120; ++i) {
        counts.push_back(pois(rng));
        features.push_back(i % 3 == 0 ? 7 : 2);
    }
    CountPosteriorModel model(counts, features, 1.0, 0);
    const PosteriorEstimate e = model.estimate();
    CHECK(e.mean.size() == counts.size());
    for (double v : e.variance) CHECK(v > 0);
    for (std::size_t i = 0; i < counts.size(); ++i) CHECK(e.expectedPositive[i] >= std::max(e.mean[i], 0.0) - 1e-12);
    const BootstrapMatrix b1 = model.bootstrap(6, 42), b2 = model.bootstrap(6, 42);
    CHECK(b1.samples() == 6 && b1.instances() == 120);
    CHECK(b1.values() == b2.values());
    CHECK_THROWS(CountPosteriorModel({1.0}, {0, 1}, 1.0, 0));
    CHECK_THROWS(CountPosteriorModel({1.0}, {0}, 0.0, 0));
}

TEST(expected_positive_part_matches_numerical_integration) {
    for (double mu : {-2.0, -0.3, 0.0, 0.7, 3.0})
        for (double var : {0.01, 0.5, 4.0}) {
            const double sd = std::sqrt(var);
            double s = 0;
            const int steps = 200000;
            const double lo = mu - 12 * sd, h = 24 * sd / steps;
            for (int k = 0; k <= steps; ++k) {
                const double u = lo + k * h, w = (k == 0 || k == steps) ? 0.5 : 1.0;
                s += w * std::max(u, 0.0) * std::exp(-0.5 * (u - mu) * (u - mu) / var) / (sd * std::sqrt(2 * M_PI)) * h;
            }
            CHECK_NEAR(RatingPosteriorModel::expectedPositivePart(mu, var), s, 1e-7);
        }
}

// normal-normal model of equation (4), computed by hand on a small example
TEST(rating_model_matches_hand_computation) {
    std::vector<RatedBusiness> b = {
        {0, {4, 5}, 9}, {0, {2}, 0}, {0, {}, 3},          // feature 0
        {1, {3, 3, 4}, 20}, {1, {5, 1}, 1},               // feature 1
    };
    RatingPosteriorModel model(b);
    const PosteriorEstimate e = model.estimate();
    // pooled within-business variance over businesses with >= 2 ratings: (0.5 + 2/3 + 8) / (1 + 2 + 1)
    const double s2 = (0.5 + 2.0 / 3.0 + 8.0) / 4.0;
    CHECK_NEAR(model.pooledVariance(), s2, 1e-12);
    // feature 0: business means 4.5 and 2 -> spread 3.125, inverse counts 1.5 over 2 businesses
    const double tau0 = std::max(0.05, 3.125 - s2 * 1.5 / 2), kappa0 = s2 / tau0, mu0 = 11.0 / 3.0;
    // feature 1: means 10/3 and 3 -> spread (1/6)^2 * 2, inverse counts 1/3 + 1/2
    const double tau1 = std::max(0.05, 2 * (1.0 / 6) * (1.0 / 6) - s2 * (1.0 / 3 + 0.5) / 2), kappa1 = s2 / tau1, mu1 = 16.0 / 5.0;
    struct Expect { double total, n, kappa, prior, checkins; };
    const Expect x[] = {{9, 2, kappa0, mu0, 9}, {2, 1, kappa0, mu0, 0}, {0, 0, kappa0, mu0, 3}, {10, 3, kappa1, mu1, 20}, {6, 2, kappa1, mu1, 1}};
    for (int i = 0; i < 5; ++i) {
        const double theta = (x[i].total + x[i].kappa * x[i].prior) / (x[i].n + x[i].kappa);
        const double w = std::log(1 + x[i].checkins);
        CHECK_NEAR(e.mean[i], w * (theta - 3) / 2, 1e-12);
        CHECK_NEAR(e.variance[i], w * w * s2 / (x[i].n + x[i].kappa) / 4, 1e-12);
    }
    CHECK(std::isnan(RatingPosteriorModel::testUtility(5, {}, RatingSettings{})));
    CHECK_NEAR(RatingPosteriorModel::testUtility(5, {4, 5}, RatingSettings{}), std::log(6.0) * 1.5 / 2, 1e-12);
    const BootstrapMatrix bm = model.bootstrap(8, 1);
    CHECK(bm.samples() == 8 && bm.instances() == 5);
    CHECK(model.bootstrap(8, 1).values() == bm.values());
}
