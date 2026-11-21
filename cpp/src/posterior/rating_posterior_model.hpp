// Rating utility of equation (4): ratings s ~ N(theta_o, sigma_s^2), theta_o ~ N(mu_f, sigma_s^2 / kappa_f);
// u(o) = w_o (theta_o - s0) / 2 with the activity weight w_o = ln(1 + a(o)) of the training check-ins.
// sigma_s^2 is the pooled within-business variance, mu_f the review-weighted mean rating of the feature,
// kappa_f = sigma_s^2 / max(tau_f^2, floor), tau_f^2 = variance of the business means minus their sampling variance.
// The bootstrap redraws every mean rating from N(mean, sigma_s^2 / n_o) and every within variance from
// sigma_s^2 chi^2_{n_o - 1} / (n_o - 1), estimates sigma_s^2, mu_f and tau_f^2 again and recomputes the posterior means.
#pragma once

#include <vector>

#include "posterior/posterior_model.hpp"

namespace colocation {

struct RatingSettings {
    double neutralRating = 3.0;     // s0, the middle of the 1-5 scale
    double tauFloor = 0.05;         // floor of tau_f^2
};

struct RatedBusiness {
    int feature = 0;
    std::vector<double> trainingStars;
    double trainingCheckins = 0;    // a(o)
};

class RatingPosteriorModel : public PosteriorModel {
public:
    RatingPosteriorModel(std::vector<RatedBusiness> businesses, RatingSettings settings = {});

    PosteriorEstimate estimate() override;
    BootstrapMatrix bootstrap(int samples, uint64_t seed) override;   // needs no prior call to estimate()

    double pooledVariance() const { return pooledVariance_; }
    // test utility of a business: ln(1 + test check-ins) (mean test rating - s0) / 2 (NaN without test ratings)
    static double testUtility(double testCheckins, const std::vector<double>& testStars, const RatingSettings& s);
    // E[u+] for u ~ N(mean, variance)
    static double expectedPositivePart(double mean, double variance);

private:
    std::vector<RatedBusiness> businesses_;
    RatingSettings settings_;
    double pooledVariance_ = 0.0;
};

}  // namespace colocation
