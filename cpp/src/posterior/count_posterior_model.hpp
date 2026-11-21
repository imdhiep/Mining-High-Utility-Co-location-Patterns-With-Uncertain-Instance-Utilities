// Count utility of Definition 4: per feature, x_o ~ Poisson(lambda_o E) (observed only when x_o >= t), prior of
// theta = ln(lambda) estimated by NPMLE and smoothed, u(o) = theta_o - c_f with the moment center c_f. The bootstrap
// redraws every count from Poisson(x_o) with the instances kept, estimates the prior and the center again, and
// recomputes the posterior means of the observed counts (Definition 5).
#pragma once

#include <random>
#include <vector>

#include "posterior/npmle.hpp"
#include "posterior/posterior_model.hpp"

namespace colocation {

class CountPosteriorModel : public PosteriorModel {
public:
    // features: any integer labels; features are processed in increasing label order
    CountPosteriorModel(std::vector<double> counts, std::vector<int> features, double exposure, int truncation,
                        NpmleSettings settings = {});

    PosteriorEstimate estimate() override;
    BootstrapMatrix bootstrap(int samples, uint64_t seed) override;
    // largest posterior mass on the edges of a local grid in the last estimate() (see LocalPosterior::edgeMass)
    double maxLocalEdgeMass() const { return maxLocalEdgeMass_; }

private:
    struct FeatureFit {                  // what the bootstrap needs for one feature
        std::vector<int> instances;
        std::vector<double> counts;
        std::vector<double> grid;
        PoissonLikelihood likelihood;
        std::vector<double> weights;     // unsmoothed NPMLE
        double center;
    };

    std::vector<double> counts_;
    std::vector<int> features_;
    double exposure_;
    int truncation_;
    NpmleSettings settings_;
    std::vector<FeatureFit> fits_;
    double maxLocalEdgeMass_ = 0.0;

    double centerOf(const std::vector<double>& counts, const std::vector<double>& grid, const std::vector<double>& weights) const;
    std::vector<double> redrawCounts(const std::vector<double>& counts, std::mt19937_64& rng) const;
};

}  // namespace colocation
