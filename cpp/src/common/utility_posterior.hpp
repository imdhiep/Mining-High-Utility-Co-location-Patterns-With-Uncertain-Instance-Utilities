// Posterior summary of the instance utilities that the miner consumes, and its binary file format.
// File layout (little endian): int64 n, int64 B, double Z, n means, n variances, B x n bootstrap posterior means
// (row major). B = 0 means no bootstrap matrix; Z <= 0 means "use the sum of the positive means".
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace colocation {

// B x n matrix of posterior means under B bootstrap estimates of the shared quantities (prior, center, ...)
class BootstrapMatrix {
public:
    BootstrapMatrix() = default;
    BootstrapMatrix(int samples, int instances) : samples_(samples), instances_(instances), values_(std::size_t(samples) * instances, 0.0) {}
    int samples() const { return samples_; }
    int instances() const { return instances_; }
    bool empty() const { return samples_ == 0; }
    double& at(int sample, int instance) { return values_[std::size_t(sample) * instances_ + instance]; }
    double at(int sample, int instance) const { return values_[std::size_t(sample) * instances_ + instance]; }
    const std::vector<double>& values() const { return values_; }
    std::vector<double>& values() { return values_; }

private:
    int samples_ = 0;
    int instances_ = 0;
    std::vector<double> values_;
};

struct UtilityPosterior {
    std::vector<double> mean;
    std::vector<double> variance;
    BootstrapMatrix bootstrap;
    double normalizer = 0.0;

    std::size_t size() const { return mean.size(); }
};

class UtilityPosteriorFile {
public:
    static UtilityPosterior read(const std::string& path);
    static void write(const std::string& path, const UtilityPosterior& posterior);
};

}  // namespace colocation
