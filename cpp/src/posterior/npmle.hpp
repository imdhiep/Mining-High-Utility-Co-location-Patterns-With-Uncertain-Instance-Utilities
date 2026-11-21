// Building blocks of the count posterior (Definition 4): a grid of log-rates, the Poisson likelihood of counts on the
// grid (optionally conditioned on x >= t), the NPMLE of the prior by EM (Kiefer and Wolfowitz; Koenker and Mizera),
// its Gaussian-kernel smoothing (bandwidth 0.9 sd n^(-1/5), Gu and Koenker), posterior moments on the grid with a local
// fine grid for counts whose likelihood is narrower than the grid, and the two centers.
#pragma once

#include <vector>

namespace colocation {

struct NpmleSettings {
    int gridPoints = 300;               // J
    int emIterations = 300;
    int bootstrapEmIterations = 40;
    double gridMarginBelow = 3.0;       // grid from ln((min x + 0.5) / E) - 3 ...
    double gridMarginAbove = 1.0;       // ... to ln((max x + 0.5) / E) + 1
    int localPoints = 201;              // local grid of a sharp instance
    double localHalfWidthSd = 7.0;      // in likelihood standard deviations 1 / sqrt(x + 0.5)
    double localStepsThreshold = 5.0;   // local grid when the likelihood sd is below this many grid steps
    double bandwidthFactor = 0.9;
    double weightFloor = 1e-300;
    double varianceFloor = 1e-12;
};

class LogRateGrid {
public:
    // numpy.linspace of J points between ln((min x + 0.5)/E) - margin and ln((max x + 0.5)/E) + margin
    static std::vector<double> forCounts(const std::vector<double>& counts, double exposure, const NpmleSettings& s);
    static std::vector<double> linspace(double lo, double hi, int points);
};

// n x J matrix L(i, j) = P(x_i | theta_j) / max_j P(x_i | theta_j), Poisson with mean E exp(theta), conditioned on x >= t
class PoissonLikelihood {
public:
    PoissonLikelihood(const std::vector<double>& counts, const std::vector<double>& grid, double exposure, int truncation);
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    const double* row(int i) const { return &values_[static_cast<std::size_t>(i) * cols_]; }
    static double logSurvival(int truncation, double mean);   // ln P(X >= t), X ~ Poisson(mean)

private:
    int rows_, cols_;
    std::vector<double> values_;
};

class Npmle {
public:
    // EM for the prior weights on the grid, starting from `weights`
    static std::vector<double> fit(const PoissonLikelihood& L, std::vector<double> weights, int iterations, double floor);
    // Gaussian-kernel smoothing; returns the weights unchanged when the bandwidth is below half a grid step
    static std::vector<double> smooth(const std::vector<double>& grid, const std::vector<double>& weights, int n,
                                      const NpmleSettings& s);
};

struct LocalPosterior {
    int instance = 0;                    // row of the likelihood matrix
    std::vector<double> grid, weights;   // local grid and normalized posterior on it
};

struct PosteriorMoments {
    std::vector<double> mean, variance;
    std::vector<double> gridPosterior;   // n x J (only when requested)
    std::vector<LocalPosterior> local;   // sharp instances (grids only when requested)
};

class GridPosterior {
public:
    static PosteriorMoments compute(const std::vector<double>& counts, double exposure, const std::vector<double>& grid,
                                    const std::vector<double>& prior, int truncation, const PoissonLikelihood& L,
                                    bool keepDistributions, const NpmleSettings& s);
};

class RateCenter {
public:
    // c = ln m - 0.5 ln(1 + v / m^2), m and v unbiased from the factorial moments of the counts (t = 0)
    static double fromCounts(const std::vector<double>& counts, double exposure);
    // c = 2 ln E[lambda] - 0.5 ln E[lambda^2] under the grid prior (t > 0)
    static double fromPrior(const std::vector<double>& grid, const std::vector<double>& weights);
};

}  // namespace colocation
