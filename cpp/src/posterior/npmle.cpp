#include "posterior/npmle.hpp"

#include <algorithm>
#include <cmath>

namespace colocation {

namespace {

double interpolate(double x, const std::vector<double>& xp, const std::vector<double>& fp) {   // numpy.interp
    if (x <= xp.front()) return fp.front();
    if (x >= xp.back()) return fp.back();
    const std::size_t k = static_cast<std::size_t>(std::upper_bound(xp.begin(), xp.end(), x) - xp.begin());
    const double t = (x - xp[k - 1]) / (xp[k] - xp[k - 1]);
    return fp[k - 1] + t * (fp[k] - fp[k - 1]);
}

}  // namespace

std::vector<double> LogRateGrid::linspace(double lo, double hi, int points) {
    std::vector<double> v(points);
    const double step = (hi - lo) / (points - 1);
    for (int i = 0; i < points; ++i) v[i] = lo + i * step;
    v[points - 1] = hi;
    return v;
}

std::vector<double> LogRateGrid::forCounts(const std::vector<double>& counts, double exposure, const NpmleSettings& s) {
    const double lo = *std::min_element(counts.begin(), counts.end());
    const double hi = *std::max_element(counts.begin(), counts.end());
    return linspace(std::log((lo + 0.5) / exposure) - s.gridMarginBelow, std::log((hi + 0.5) / exposure) + s.gridMarginAbove,
                    s.gridPoints);
}

double PoissonLikelihood::logSurvival(int truncation, double mean) {
    if (truncation <= 0) return 0.0;
    double p;
    if (truncation == 1) {
        p = -std::expm1(-mean);
    } else {
        double term = std::exp(-mean), cdf = term;
        for (int i = 1; i < truncation; ++i) {
            term *= mean / i;
            cdf += term;
        }
        p = 1.0 - cdf;
    }
    return std::log(std::max(p, 1e-300));
}

PoissonLikelihood::PoissonLikelihood(const std::vector<double>& counts, const std::vector<double>& grid, double exposure,
                                     int truncation)
    : rows_(static_cast<int>(counts.size())), cols_(static_cast<int>(grid.size())),
      values_(static_cast<std::size_t>(counts.size()) * grid.size()) {
    std::vector<double> mean(cols_), logSf(cols_, 0.0);
    for (int j = 0; j < cols_; ++j) {
        mean[j] = exposure * std::exp(grid[j]);
        if (truncation > 0) logSf[j] = logSurvival(truncation, mean[j]);
    }
    for (int i = 0; i < rows_; ++i) {
        double* r = &values_[static_cast<std::size_t>(i) * cols_];
        const double logFactorial = std::lgamma(counts[i] + 1);
        double peak = -INFINITY;
        for (int j = 0; j < cols_; ++j) {
            r[j] = counts[i] * grid[j] - mean[j] - logFactorial - logSf[j];
            peak = std::max(peak, r[j]);
        }
        for (int j = 0; j < cols_; ++j) r[j] = std::exp(r[j] - peak);
    }
}

std::vector<double> Npmle::fit(const PoissonLikelihood& L, std::vector<double> w, int iterations, double floor) {
    const int n = L.rows(), J = L.cols();
    std::vector<double> sum(J), post(J);
    for (int it = 0; it < iterations; ++it) {
        std::fill(sum.begin(), sum.end(), 0.0);
        for (int i = 0; i < n; ++i) {
            const double* r = L.row(i);
            double total = 0;
            for (int j = 0; j < J; ++j) {
                post[j] = r[j] * w[j];
                total += post[j];
            }
            for (int j = 0; j < J; ++j) sum[j] += post[j] / total;
        }
        for (int j = 0; j < J; ++j) w[j] = std::max(sum[j] / n, floor);
    }
    return w;
}

std::vector<double> Npmle::smooth(const std::vector<double>& grid, const std::vector<double>& w, int n, const NpmleSettings& s) {
    const int J = static_cast<int>(grid.size());
    double m = 0, m2 = 0;
    for (int j = 0; j < J; ++j) {
        m += w[j] * grid[j];
        m2 += w[j] * grid[j] * grid[j];
    }
    const double sd = std::sqrt(std::max(m2 - m * m, s.varianceFloor));
    const double h = s.bandwidthFactor * sd * std::pow(static_cast<double>(n), -0.2);
    if (h <= 0.5 * (grid[1] - grid[0])) return w;
    std::vector<double> out(J, 0.0);
    double total = 0;
    for (int i = 0; i < J; ++i) {
        double acc = 0;
        for (int j = 0; j < J; ++j) {
            const double z = (grid[i] - grid[j]) / h;
            acc += std::exp(-0.5 * z * z) * w[j];
        }
        out[i] = acc;
        total += acc;
    }
    for (double& v : out) v /= total;
    return out;
}

PosteriorMoments GridPosterior::compute(const std::vector<double>& x, double E, const std::vector<double>& grid,
                                        const std::vector<double>& prior, int t, const PoissonLikelihood& L,
                                        bool keep, const NpmleSettings& s) {
    const int n = static_cast<int>(x.size()), J = static_cast<int>(grid.size());
    PosteriorMoments out;
    out.mean.resize(n);
    out.variance.resize(n);
    if (keep) out.gridPosterior.resize(static_cast<std::size_t>(n) * J);
    std::vector<double> p(J);
    for (int i = 0; i < n; ++i) {
        const double* r = L.row(i);
        double total = 0;
        for (int j = 0; j < J; ++j) {
            p[j] = r[j] * prior[j];
            total += p[j];
        }
        double m = 0, m2 = 0;
        for (int j = 0; j < J; ++j) {
            p[j] /= total;
            m += p[j] * grid[j];
            m2 += p[j] * grid[j] * grid[j];
        }
        out.mean[i] = m;
        out.variance[i] = std::max(m2 - m * m, s.varianceFloor);
        if (keep) std::copy(p.begin(), p.end(), out.gridPosterior.begin() + static_cast<std::ptrdiff_t>(i) * J);
    }
    // instances whose likelihood is narrower than localStepsThreshold grid steps: local fine grid
    const double step = grid[1] - grid[0];
    std::vector<double> logPrior(J);
    for (int j = 0; j < J; ++j) logPrior[j] = std::log(std::max(prior[j], s.weightFloor));
    const std::vector<double> offsets = LogRateGrid::linspace(-s.localHalfWidthSd, s.localHalfWidthSd, s.localPoints);
    for (int i = 0; i < n; ++i) {
        const double sd = 1.0 / std::sqrt(x[i] + 0.5);
        if (!(sd < s.localStepsThreshold * step)) continue;
        LocalPosterior lp;
        lp.instance = i;
        lp.grid.resize(s.localPoints);
        lp.weights.resize(s.localPoints);
        std::vector<double> logPost(s.localPoints);
        const double centre = std::log((x[i] + 0.5) / E);
        double peak = -INFINITY;
        for (int k = 0; k < s.localPoints; ++k) {
            lp.grid[k] = centre + sd * offsets[k];
            logPost[k] = x[i] * lp.grid[k] - E * std::exp(lp.grid[k]) + interpolate(lp.grid[k], grid, logPrior);
            if (t > 0) logPost[k] -= PoissonLikelihood::logSurvival(t, E * std::exp(lp.grid[k]));
            peak = std::max(peak, logPost[k]);
        }
        double total = 0;
        for (int k = 0; k < s.localPoints; ++k) {
            lp.weights[k] = std::exp(logPost[k] - peak);
            total += lp.weights[k];
        }
        double m = 0, m2 = 0;
        for (int k = 0; k < s.localPoints; ++k) {
            lp.weights[k] /= total;
            m += lp.weights[k] * lp.grid[k];
            m2 += lp.weights[k] * lp.grid[k] * lp.grid[k];
        }
        out.mean[i] = m;
        out.variance[i] = std::max(m2 - m * m, s.varianceFloor);
        if (!keep) {
            lp.grid.clear();
            lp.weights.clear();
        }
        out.local.push_back(std::move(lp));
    }
    return out;
}

double RateCenter::fromCounts(const std::vector<double>& x, double E) {
    const double n = static_cast<double>(x.size());
    double sum = 0, factorial2 = 0;
    for (double v : x) {
        sum += v;
        factorial2 += v * (v - 1);
    }
    double m = sum / n / E;
    const double var = std::max(factorial2 / n / (E * E) - m * m + m / (n * E), 0.0);
    m = std::max(m, 1e-12);
    return std::log(m) - 0.5 * std::log1p(var / (m * m));
}

double RateCenter::fromPrior(const std::vector<double>& grid, const std::vector<double>& w) {
    double first = 0, second = 0;
    for (std::size_t j = 0; j < grid.size(); ++j) {
        const double rate = std::exp(grid[j]);
        first += w[j] * rate;
        second += w[j] * rate * rate;
    }
    return 2 * std::log(first) - 0.5 * std::log(second);
}

}  // namespace colocation
