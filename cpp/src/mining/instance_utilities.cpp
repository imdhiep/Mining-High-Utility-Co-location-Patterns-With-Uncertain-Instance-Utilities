#include "mining/instance_utilities.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace colocation {

InstanceUtilities::InstanceUtilities(const UtilityPosterior& p)
    : mean_(p.mean), variance_(p.variance), samples_(p.bootstrap.samples()) {
    if (mean_.size() != variance_.size()) throw std::invalid_argument("means and variances differ in size");
    if (samples_ > 0 && p.bootstrap.instances() != size()) throw std::invalid_argument("bootstrap matrix size");
    buildDeviations(p.bootstrap);
    if (p.normalizer > 0) {
        normalizer_ = p.normalizer;
    } else {
        for (double m : mean_)
            if (m > 0) normalizer_ += m;
    }
    buildRankOrder();
}

void InstanceUtilities::buildDeviations(const BootstrapMatrix& bootstrap) {
    const int n = size();
    deviations_.assign(static_cast<std::size_t>(samples_) * n, 0.0);
    if (samples_ == 0) return;
    const double scale = 1.0 / std::sqrt(static_cast<double>(std::max(samples_ - 1, 1)));
    for (int i = 0; i < n; ++i) {
        double average = 0;
        for (int b = 0; b < samples_; ++b) average += bootstrap.at(b, i);
        average /= samples_;
        for (int b = 0; b < samples_; ++b) deviations_[static_cast<std::size_t>(b) * n + i] = (bootstrap.at(b, i) - average) * scale;
    }
}

// positive-mean instances by increasing variance / mean, compared exactly by cross-multiplication; ties keep the
// instance order
void InstanceUtilities::buildRankOrder() {
    std::vector<int> positive;
    for (int i = 0; i < size(); ++i)
        if (mean_[i] > 0) positive.push_back(i);
    std::stable_sort(positive.begin(), positive.end(),
                     [&](int a, int b) { return variance_[a] * mean_[b] < variance_[b] * mean_[a]; });
    rank_.assign(size(), -1);
    rankedMean_.resize(positive.size());
    rankedVariance_.resize(positive.size());
    for (std::size_t r = 0; r < positive.size(); ++r) {
        rank_[positive[r]] = static_cast<int>(r);
        rankedMean_[r] = mean_[positive[r]];
        rankedVariance_[r] = variance_[positive[r]];
    }
}

double InstanceUtilities::score(const std::vector<int>& instances, double z) const {
    double total = 0, var = 0;
    for (int a : instances) {
        total += mean_[a];
        var += variance_[a];
    }
    const int n = size();
    for (int b = 0; b < samples_; ++b) {
        const double* row = &deviations_[static_cast<std::size_t>(b) * n];
        double shared = 0;
        for (int a : instances) shared += row[a];
        var += shared * shared;
    }
    return (total - z * std::sqrt(std::max(var, 0.0))) / normalizer_;
}

}  // namespace colocation
