// Set rule S_delta of Section IV-C: among candidate patterns with total mean M and total variance V, the probability
// that a pattern reaches the threshold is p = Phi((M - threshold) / sqrt(V)); the rule keeps the largest prefix of the
// candidates in decreasing p (ties: candidate order) whose mean of (1 - p) is at most delta, i.e. whose expected share
// of patterns below the threshold is at most delta under the posterior.
#pragma once

#include <vector>

namespace colocation {

struct SetSelection {
    std::vector<bool> selected;        // one per candidate
    std::vector<double> probability;   // p of every candidate
};

class SetRule {
public:
    static SetSelection select(const std::vector<double>& totalMean, const std::vector<double>& totalVariance, double threshold,
                               double delta);
};

}  // namespace colocation
