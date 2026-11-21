// Posterior means and variances of the instance utilities as the miner uses them (Sections III-B and IV-A):
//   score of a set A of participating instances: (sum_A mu - z sqrt(V(A))) / Z,
//   V(A) = sum_A sigma^2 + sum_b (sum_A D[b, a])^2, D = bootstrap posterior means centred per instance and divided by
//   sqrt(B - 1) (the shared-error term of the total variance V_T), and the order of the positive-mean instances by
//   increasing sigma^2 / mu used by the subtree bounds (Lemma 1).
#pragma once

#include <vector>

#include "common/utility_posterior.hpp"

namespace colocation {

class InstanceUtilities {
public:
    explicit InstanceUtilities(const UtilityPosterior& posterior);

    int size() const { return static_cast<int>(mean_.size()); }
    double mean(int i) const { return mean_[i]; }
    double variance(int i) const { return variance_[i]; }
    double normalizer() const { return normalizer_; }
    int bootstrapSamples() const { return samples_; }

    // score of the set of participating instances A (distinct ids)
    double score(const std::vector<int>& instances, double z) const;

    // rank of a positive-mean instance in the order of increasing variance / mean, -1 for the others
    int rank(int i) const { return rank_[i]; }
    int positiveCount() const { return static_cast<int>(rankedMean_.size()); }
    double rankedMean(int r) const { return rankedMean_[r]; }
    double rankedVariance(int r) const { return rankedVariance_[r]; }

private:
    std::vector<double> mean_, variance_;
    std::vector<double> deviations_;   // samples_ x n, centred and scaled bootstrap means
    int samples_ = 0;
    double normalizer_ = 0.0;
    std::vector<int> rank_;
    std::vector<double> rankedMean_, rankedVariance_;

    void buildDeviations(const BootstrapMatrix& bootstrap);
    void buildRankOrder();
};

}  // namespace colocation
