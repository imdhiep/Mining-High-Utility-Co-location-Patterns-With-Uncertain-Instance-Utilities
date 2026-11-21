#include "posterior/posterior_model.hpp"

namespace colocation {

double PosteriorEstimate::expectedNormalizer() const {
    double z = 0;
    for (double e : expectedPositive) z += e;
    return z;
}

UtilityPosterior PosteriorModel::summarize(int samples, uint64_t seed) {
    const PosteriorEstimate est = estimate();
    UtilityPosterior out;
    out.mean = est.mean;
    out.variance = est.variance;
    out.normalizer = est.expectedNormalizer();
    if (samples > 0) out.bootstrap = bootstrap(samples, seed);
    return out;
}

}  // namespace colocation
