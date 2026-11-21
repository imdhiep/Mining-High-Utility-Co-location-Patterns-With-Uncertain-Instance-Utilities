#include "posterior/count_posterior_model.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace colocation {

CountPosteriorModel::CountPosteriorModel(std::vector<double> counts, std::vector<int> features, double exposure, int truncation,
                                         NpmleSettings settings, CenterRule center)
    : counts_(std::move(counts)), features_(std::move(features)), exposure_(exposure), truncation_(truncation),
      settings_(settings), center_(center) {
    if (counts_.size() != features_.size()) throw std::invalid_argument("counts and features differ in size");
    if (exposure_ <= 0) throw std::invalid_argument("exposure must be positive");
}

double CountPosteriorModel::centerOf(const std::vector<double>& counts, const std::vector<double>& grid,
                                     const std::vector<double>& weights) const {
    if (center_ == CenterRule::PriorMedian) return RateCenter::priorMedian(grid, weights);
    return truncation_ == 0 ? RateCenter::fromCounts(counts, exposure_) : RateCenter::fromPrior(grid, weights);
}

PosteriorEstimate CountPosteriorModel::estimate() {
    const int n = static_cast<int>(counts_.size());
    PosteriorEstimate est;
    est.mean.assign(n, 0.0);
    est.variance.assign(n, 0.0);
    est.expectedPositive.assign(n, 0.0);
    fits_.clear();
    maxLocalEdgeMass_ = 0.0;
    const std::set<int> labels(features_.begin(), features_.end());
    for (int label : labels) {
        std::vector<int> idx;
        std::vector<double> x;
        for (int i = 0; i < n; ++i)
            if (features_[i] == label) {
                idx.push_back(i);
                x.push_back(counts_[i]);
            }
        std::vector<double> grid = LogRateGrid::forCounts(x, exposure_, settings_);
        PoissonLikelihood L(x, grid, exposure_, truncation_);
        std::vector<double> w = Npmle::fit(L, std::vector<double>(grid.size(), 1.0 / grid.size()), settings_.emIterations,
                                           settings_.weightFloor);
        const std::vector<double> smoothed = Npmle::smooth(grid, w, static_cast<int>(x.size()), settings_);
        const double c = centerOf(x, grid, w);
        const PosteriorMoments pm = GridPosterior::compute(x, exposure_, grid, smoothed, truncation_, L, true, settings_);
        const int J = static_cast<int>(grid.size());
        for (std::size_t a = 0; a < x.size(); ++a) {
            double e = 0;
            for (int j = 0; j < J; ++j) e += pm.gridPosterior[a * J + j] * std::max(grid[j] - c, 0.0);
            est.expectedPositive[idx[a]] = e;
        }
        for (const LocalPosterior& lp : pm.local) {
            maxLocalEdgeMass_ = std::max(maxLocalEdgeMass_, lp.edgeMass);
            double e = 0;
            for (std::size_t k = 0; k < lp.grid.size(); ++k) e += lp.weights[k] * std::max(lp.grid[k] - c, 0.0);
            est.expectedPositive[idx[lp.instance]] = e;
        }
        for (std::size_t a = 0; a < x.size(); ++a) {
            est.mean[idx[a]] = pm.mean[a] - c;
            est.variance[idx[a]] = pm.variance[a];
        }
        fits_.push_back({std::move(idx), std::move(x), std::move(grid), std::move(L), std::move(w), c});
    }
    return est;
}

// x* ~ Poisson(x), redrawn while below the inclusion threshold t
std::vector<double> CountPosteriorModel::redrawCounts(const std::vector<double>& counts, std::mt19937_64& rng) const {
    auto draw = [&](double mean) -> double {
        if (mean <= 0) return 0.0;
        std::poisson_distribution<long long> poisson(mean);
        return static_cast<double>(poisson(rng));
    };
    std::vector<double> out(counts.size());
    for (std::size_t a = 0; a < counts.size(); ++a) out[a] = draw(counts[a]);
    if (truncation_ > 0) {
        bool redraw = true;
        while (redraw) {
            redraw = false;
            for (std::size_t a = 0; a < counts.size(); ++a)
                if (out[a] < truncation_) {
                    out[a] = draw(counts[a]);
                    redraw = redraw || out[a] < truncation_;
                }
        }
    }
    return out;
}

BootstrapMatrix CountPosteriorModel::bootstrap(int samples, uint64_t seed) {
    if (fits_.empty()) throw std::logic_error("bootstrap before estimate");
    BootstrapMatrix U(samples, static_cast<int>(counts_.size()));
    std::mt19937_64 rng(seed);
    for (const FeatureFit& fit : fits_) {
        const int nf = static_cast<int>(fit.counts.size());
        for (int b = 0; b < samples; ++b) {
            const std::vector<double> redrawn = redrawCounts(fit.counts, rng);
            const PoissonLikelihood Lb(redrawn, fit.grid, exposure_, truncation_);
            const std::vector<double> w = Npmle::fit(Lb, fit.weights, settings_.bootstrapEmIterations, settings_.weightFloor);
            const PosteriorMoments pm = GridPosterior::compute(fit.counts, exposure_, fit.grid, Npmle::smooth(fit.grid, w, nf, settings_),
                                                               truncation_, fit.likelihood, false, settings_);
            const double c = centerOf(redrawn, fit.grid, w);
            for (int a = 0; a < nf; ++a) U.at(b, fit.instances[a]) = pm.mean[a] - c;
        }
    }
    return U;
}

}  // namespace colocation
