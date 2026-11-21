// Common interface of the posterior models of Section IV-A: each model gives every instance a posterior mean and
// variance of its utility and E[u+ | data] (whose sum is the expected normalizer Z), and a bootstrap matrix of posterior
// means under re-estimated shared quantities (the second term of the total variance V_T).
#pragma once

#include <cstdint>
#include <vector>

#include "common/utility_posterior.hpp"

namespace colocation {

struct PosteriorEstimate {
    std::vector<double> mean;
    std::vector<double> variance;
    std::vector<double> expectedPositive;   // E[u+ | data]

    double expectedNormalizer() const;
};

class PosteriorModel {
public:
    virtual ~PosteriorModel() = default;
    virtual PosteriorEstimate estimate() = 0;
    // requires estimate() first; B x n posterior means of the observed evidence under re-estimated shared quantities
    virtual BootstrapMatrix bootstrap(int samples, uint64_t seed) = 0;

    // estimate, bootstrap and pack them for the miner
    UtilityPosterior summarize(int samples, uint64_t seed);
};

}  // namespace colocation
