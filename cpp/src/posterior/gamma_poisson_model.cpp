#include "posterior/gamma_poisson_model.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

#include "posterior/npmle.hpp"
#include "stats/special_functions.hpp"

namespace colocation {

GammaPoissonPosteriorModel::GammaPoissonPosteriorModel(std::vector<double> counts, std::vector<int> features, double exposure)
    : counts_(std::move(counts)), features_(std::move(features)), exposure_(exposure) {
    if (counts_.size() != features_.size()) throw std::invalid_argument("counts and features differ in size");
    if (exposure_ <= 0) throw std::invalid_argument("exposure must be positive");
}

std::vector<std::vector<int>> GammaPoissonPosteriorModel::groups() const {
    std::map<int, std::vector<int>> byLabel;
    for (std::size_t i = 0; i < features_.size(); ++i) byLabel[features_[i]].push_back(static_cast<int>(i));
    std::vector<std::vector<int>> out;
    for (auto& kv : byLabel) out.push_back(std::move(kv.second));
    return out;
}

GammaPoissonPosteriorModel::Prior GammaPoissonPosteriorModel::priorOf(const std::vector<double>& x) const {
    const double n = static_cast<double>(x.size());
    double mean = 0;
    for (double v : x) mean += v / n;
    double var = 0;
    for (double v : x) var += (v - mean) * (v - mean);
    var /= std::max(n - 1, 1.0);
    const double shape = var > mean ? mean * mean / (var - mean) : kNoOverdispersion;
    return {shape, shape * exposure_ / std::max(mean, 1e-9)};
}

double GammaPoissonPosteriorModel::expectedPositivePart(double shape, double rate, double center) {
    // density of theta: rate^shape / Gamma(shape) exp(shape theta - rate e^theta)
    const double mean = SpecialFunctions::digamma(shape) - std::log(rate), sd = std::sqrt(SpecialFunctions::trigamma(shape));
    const std::vector<double> grid = LogRateGrid::linspace(mean - kIntegrationHalfWidthSd * sd, mean + kIntegrationHalfWidthSd * sd,
                                                           kIntegrationPoints);
    const double logNorm = shape * std::log(rate) - std::lgamma(shape);
    const double step = grid[1] - grid[0];
    double total = 0, mass = 0;
    for (std::size_t k = 0; k < grid.size(); ++k) {
        const double w = std::exp(logNorm + shape * grid[k] - rate * std::exp(grid[k])) * step;
        mass += w;
        total += w * std::max(grid[k] - center, 0.0);
    }
    return total / mass;
}

PosteriorEstimate GammaPoissonPosteriorModel::estimate() {
    const std::size_t n = counts_.size();
    PosteriorEstimate est;
    est.mean.assign(n, 0.0);
    est.variance.assign(n, 0.0);
    est.expectedPositive.assign(n, 0.0);
    for (const auto& idx : groups()) {
        std::vector<double> x;
        for (int i : idx) x.push_back(counts_[i]);
        const Prior p = priorOf(x);
        const double c = RateCenter::fromCounts(x, exposure_);
        for (std::size_t a = 0; a < idx.size(); ++a) {
            const double shape = p.shape + x[a], rate = p.rate + exposure_;
            est.mean[idx[a]] = SpecialFunctions::digamma(shape) - std::log(rate) - c;
            est.variance[idx[a]] = SpecialFunctions::trigamma(shape);
            est.expectedPositive[idx[a]] = expectedPositivePart(shape, rate, c);
        }
    }
    return est;
}

BootstrapMatrix GammaPoissonPosteriorModel::bootstrap(int samples, uint64_t seed) {
    BootstrapMatrix U(samples, static_cast<int>(counts_.size()));
    std::mt19937_64 rng(seed);
    for (const auto& idx : groups()) {
        std::vector<double> x;
        for (int i : idx) x.push_back(counts_[i]);
        for (int b = 0; b < samples; ++b) {
            std::vector<double> redrawn(x.size());
            for (std::size_t a = 0; a < x.size(); ++a) {
                if (x[a] <= 0) continue;
                std::poisson_distribution<long long> poisson(x[a]);
                redrawn[a] = static_cast<double>(poisson(rng));
            }
            const Prior p = priorOf(redrawn);
            const double c = RateCenter::fromCounts(redrawn, exposure_);
            for (std::size_t a = 0; a < idx.size(); ++a)
                U.at(b, idx[a]) = SpecialFunctions::digamma(p.shape + x[a]) - std::log(p.rate + exposure_) - c;
        }
    }
    return U;
}

}  // namespace colocation
