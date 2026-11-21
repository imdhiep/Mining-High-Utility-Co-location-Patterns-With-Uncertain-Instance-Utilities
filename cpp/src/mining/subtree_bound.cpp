#include "mining/subtree_bound.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace colocation {

BoundTest parseBoundTest(const std::string& name) {
    if (name == "two") return BoundTest::MeanChordPrefix;
    if (name == "exact") return BoundTest::MeanPrefix;
    if (name == "mean") return BoundTest::MeanOnly;
    if (name == "none") return BoundTest::None;
    throw std::invalid_argument("unknown bound test " + name);
}

double SubtreeBound::prefixScan(const std::vector<double>& means, const std::vector<double>& variances, double z) {
    double total = 0, var = 0, best = 0;
    for (std::size_t i = 0; i < means.size(); ++i) {
        total += means[i];
        var += variances[i];
        best = std::max(best, total - z * std::sqrt(var));
    }
    return best;
}

double SubtreeBound::chordBound(const std::vector<double>& means, const std::vector<double>& variances, double z) {
    double sumVar = 0, sumMean = 0;
    for (double v : variances) sumVar += v;
    for (double m : means) sumMean += m;
    if (sumVar <= 0) return sumMean;
    const double root = std::sqrt(sumVar);
    double bound = 0;
    for (std::size_t i = 0; i < means.size(); ++i) bound += std::max(means[i] - z * variances[i] / root, 0.0);
    return bound;
}

bool SubtreeBound::below(const std::vector<double>& means, const std::vector<double>& variances, double threshold) const {
    if (means.empty()) return 0.0 < threshold;
    double sumMean = 0;
    for (double m : means) sumMean += m;
    if (sumMean < threshold) return true;            // the mean bound already prunes
    if (meanOnly()) return false;
    if (test_ == BoundTest::MeanChordPrefix && chordBound(means, variances, z_) < threshold) return true;
    return prefixScan(means, variances, z_) < threshold;
}

double SubtreeBound::value(const std::vector<double>& means, const std::vector<double>& variances) const {
    if (meanOnly()) {
        double s = 0;
        for (double m : means) s += m;
        return s;
    }
    return prefixScan(means, variances, z_);
}

}  // namespace colocation
