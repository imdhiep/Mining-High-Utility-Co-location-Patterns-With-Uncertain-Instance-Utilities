#include "posterior/rating_posterior_model.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>

namespace colocation {

namespace {

double sumOf(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return s;
}

struct FeaturePrior {
    double mean = 0;     // mu_f
    double kappa = 0;    // kappa_f
};

// sigma_s^2: within-business squared deviations over businesses with at least two ratings, divided by sum (n - 1)
double pooledWithinVariance(const std::vector<RatedBusiness>& businesses) {
    double num = 0, den = 0;
    for (const auto& b : businesses) {
        const auto& v = b.trainingStars;
        if (v.size() < 2) continue;
        const double m = sumOf(v) / v.size();
        double squares = 0;
        for (double x : v) squares += (x - m) * (x - m);
        num += squares;
        den += static_cast<double>(v.size()) - 1;
    }
    return num / den;
}

std::map<int, FeaturePrior> featurePriors(const std::vector<RatedBusiness>& businesses, double sigma2, double tauFloor) {
    std::map<int, std::vector<const RatedBusiness*>> byFeature;
    for (const auto& b : businesses)
        if (!b.trainingStars.empty()) byFeature[b.feature].push_back(&b);
    std::map<int, FeaturePrior> priors;
    for (const auto& kv : byFeature) {
        double totalStars = 0, totalCount = 0, inverseCounts = 0;
        std::vector<double> means;
        for (const RatedBusiness* b : kv.second) {
            const double s = sumOf(b->trainingStars);
            totalStars += s;
            totalCount += b->trainingStars.size();
            means.push_back(s / b->trainingStars.size());
            inverseCounts += 1.0 / b->trainingStars.size();
        }
        const double meanOfMeans = sumOf(means) / means.size();
        double spread = 0;
        for (double x : means) spread += (x - meanOfMeans) * (x - meanOfMeans);
        spread /= std::max<double>(1, static_cast<double>(means.size()) - 1);
        const double tau2 = std::max(tauFloor, spread - sigma2 * inverseCounts / kv.second.size());
        priors[kv.first] = {totalStars / totalCount, sigma2 / tau2};
    }
    return priors;
}

}  // namespace

RatingPosteriorModel::RatingPosteriorModel(std::vector<RatedBusiness> businesses, RatingSettings settings)
    : businesses_(std::move(businesses)), settings_(settings) {}

double RatingPosteriorModel::expectedPositivePart(double mean, double variance) {
    const double sd = std::sqrt(std::max(variance, 1e-300)), z = mean / sd;
    const double cdf = 0.5 * std::erfc(-z / std::sqrt(2.0));
    const double pdf = std::exp(-0.5 * z * z) / std::sqrt(2 * M_PI);
    return mean * cdf + sd * pdf;
}

double RatingPosteriorModel::testUtility(double testCheckins, const std::vector<double>& testStars, const RatingSettings& s) {
    if (testStars.empty()) return NAN;
    return std::log1p(testCheckins) * (sumOf(testStars) / testStars.size() - s.neutralRating) / 2;
}

PosteriorEstimate RatingPosteriorModel::estimate() {
    pooledVariance_ = pooledWithinVariance(businesses_);
    const std::map<int, FeaturePrior> priors = featurePriors(businesses_, pooledVariance_, settings_.tauFloor);
    PosteriorEstimate est;
    for (const auto& b : businesses_) {
        const FeaturePrior& p = priors.at(b.feature);   // the data loader keeps only features with rated businesses
        const double n = static_cast<double>(b.trainingStars.size());
        const double theta = (sumOf(b.trainingStars) + p.kappa * p.mean) / (n + p.kappa);
        const double thetaVar = pooledVariance_ / (n + p.kappa);
        const double w = std::log1p(b.trainingCheckins);
        const double mean = w * (theta - settings_.neutralRating) / 2, var = w * w * thetaVar / 4;
        est.mean.push_back(mean);
        est.variance.push_back(var);
        est.expectedPositive.push_back(expectedPositivePart(mean, var));
    }
    return est;
}

BootstrapMatrix RatingPosteriorModel::bootstrap(int samples, uint64_t seed) {
    const int n = static_cast<int>(businesses_.size());
    std::vector<double> count(n), total(n), mean(n), within(n), dof(n), weight(n);
    double dofSum = 0, weighted = 0;
    for (int i = 0; i < n; ++i) {
        const auto& v = businesses_[i].trainingStars;
        count[i] = static_cast<double>(v.size());
        total[i] = sumOf(v);
        double squares = 0;
        for (double x : v) squares += x * x;
        mean[i] = count[i] > 0 ? total[i] / std::max(count[i], 1.0) : 0.0;
        within[i] = count[i] >= 2 ? (squares - count[i] * mean[i] * mean[i]) / std::max(count[i] - 1, 1.0) : 0.0;
        dof[i] = std::max(count[i] - 1, 0.0);
        dofSum += dof[i];
        weighted += dof[i] * within[i];
        weight[i] = std::log1p(businesses_[i].trainingCheckins);
    }
    const double sigma2 = weighted / dofSum;
    const std::set<int> features = [&] {
        std::set<int> s;
        for (const auto& b : businesses_) s.insert(b.feature);
        return s;
    }();
    BootstrapMatrix U(samples, n);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> standardNormal(0.0, 1.0);
    for (int b = 0; b < samples; ++b) {
        std::vector<double> redrawnMean(n), redrawnWithin(n);
        double redrawnWeighted = 0;
        for (int i = 0; i < n; ++i) {
            redrawnMean[i] = count[i] > 0 ? mean[i] + std::sqrt(sigma2 / std::max(count[i], 1.0)) * standardNormal(rng) : 0.0;
            if (count[i] >= 2) {
                std::chi_squared_distribution<double> chi2(std::max(dof[i], 1.0));
                redrawnWithin[i] = sigma2 * chi2(rng) / std::max(dof[i], 1.0);
            }
            redrawnWeighted += dof[i] * redrawnWithin[i];
        }
        const double sigma2b = redrawnWeighted / dofSum;
        for (int f : features) {
            double weightedCount = 0, weightedMean = 0, rated = 0, inverseCounts = 0, meanSum = 0;
            for (int i = 0; i < n; ++i)
                if (businesses_[i].feature == f && count[i] > 0) {
                    weightedCount += count[i];
                    weightedMean += count[i] * redrawnMean[i];
                    rated += 1;
                    inverseCounts += 1.0 / count[i];
                    meanSum += redrawnMean[i];
                }
            const double mu = rated > 0 ? weightedMean / weightedCount : settings_.neutralRating;
            double spread = 0;
            if (rated > 1) {
                const double average = meanSum / rated;
                for (int i = 0; i < n; ++i)
                    if (businesses_[i].feature == f && count[i] > 0) spread += (redrawnMean[i] - average) * (redrawnMean[i] - average);
                spread /= rated - 1;
            }
            const double tau2 = rated > 0 ? std::max(settings_.tauFloor, spread - sigma2b * (inverseCounts / rated)) : settings_.tauFloor;
            const double kappa = sigma2b / tau2;
            for (int i = 0; i < n; ++i)
                if (businesses_[i].feature == f) {
                    const double theta = (total[i] + kappa * mu) / (count[i] + kappa);
                    U.at(b, i) = weight[i] * (theta - settings_.neutralRating) / 2;
                }
        }
    }
    return U;
}

}  // namespace colocation
