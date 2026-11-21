// Parametric alternative to the count posterior of Definition 4, used only to show what the nonparametric prior adds:
// per feature, lambda ~ Gamma(A, beta) with A and beta from the mean m and variance v of the counts
// (A = m^2 / (v - m) when v > m, else kNoOverdispersion; beta = A E / m), so that theta = ln(lambda) has posterior mean
// digamma(A + x) - ln(beta + E) and variance trigamma(A + x). The utility is theta minus the moment center of the counts
// (the same center as Definition 4). Counts are fully observed (t = 0).
// The bootstrap redraws the counts from Poisson(x), estimates A, beta and the center again and recomputes the
// posterior means of the observed counts.
#pragma once

#include <random>
#include <vector>

#include "posterior/posterior_model.hpp"

namespace colocation {

class GammaPoissonPosteriorModel : public PosteriorModel {
public:
    static constexpr double kNoOverdispersion = 1e6;   // shape when the counts show no overdispersion
    static constexpr int kIntegrationPoints = 801;     // grid for E[u+] on theta
    static constexpr double kIntegrationHalfWidthSd = 10.0;

    GammaPoissonPosteriorModel(std::vector<double> counts, std::vector<int> features, double exposure);

    PosteriorEstimate estimate() override;
    BootstrapMatrix bootstrap(int samples, uint64_t seed) override;

private:
    struct Prior {
        double shape, rate;
    };
    std::vector<double> counts_;
    std::vector<int> features_;
    double exposure_;

    std::vector<std::vector<int>> groups() const;   // instances of each feature, features in increasing label order
    Prior priorOf(const std::vector<double>& x) const;
    // E[(theta - c)+] for theta = ln(lambda), lambda ~ Gamma(shape, rate)
    static double expectedPositivePart(double shape, double rate, double center);
};

}  // namespace colocation
