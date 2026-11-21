// Upper bounds on (sum mu - z sqrt(sum sigma^2)) over all subsets of a candidate set of instances (Section IV-B).
// The candidate set is given as positive-mean instances in rank order (increasing sigma^2 / mu), so the exact bound
// SR_z is the best prefix (Lemma 1). Two cheaper upper bounds are tried first: the mean bound (sum of the means) and
// the chord bound (sum of (mu - z sigma^2 / sqrt(V))^+ with V the total variance of the set).
#pragma once

#include <string>
#include <vector>

namespace colocation {

enum class BoundTest {
    MeanChordPrefix,   // mean bound, then chord bound, then prefix scan (the paper's setting)
    MeanPrefix,        // mean bound, then prefix scan
    MeanOnly,          // mean bound only (valid, weaker)
    None               // no pruning
};

BoundTest parseBoundTest(const std::string& name);   // "two", "exact", "mean", "none"

class SubtreeBound {
public:
    SubtreeBound(BoundTest test, double z) : test_(test), z_(z) {}

    // exact bound over all subsets of a rank-ordered candidate set: max(0, best prefix value)
    static double prefixScan(const std::vector<double>& means, const std::vector<double>& variances, double z);
    // sum of (mu - z sigma^2 / sqrt(V))^+, V = sum of the variances; equals the sum of the means when V = 0
    static double chordBound(const std::vector<double>& means, const std::vector<double>& variances, double z);

    // true when the bound of the candidate set is below the threshold (the decision is always that of SR_z, except
    // for MeanOnly, which decides with the mean bound)
    bool below(const std::vector<double>& means, const std::vector<double>& variances, double threshold) const;
    // the bound value used for the feature bounds (mean bound when MeanOnly or z <= 0, SR_z otherwise)
    double value(const std::vector<double>& means, const std::vector<double>& variances) const;

    bool prunes() const { return test_ != BoundTest::None; }
    bool meanOnly() const { return test_ == BoundTest::MeanOnly || z_ <= 0; }

private:
    BoundTest test_;
    double z_;
};

}  // namespace colocation
