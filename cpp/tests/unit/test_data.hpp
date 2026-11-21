// Random small inputs shared by the tests, and brute-force references that use no code of the library beyond the
// data types.
#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "common/utility_posterior.hpp"
#include "spatial/spatial_dataset.hpp"

namespace unit {

inline colocation::SpatialDataset randomDataset(std::mt19937_64& rng, int n, int features, double side) {
    std::uniform_int_distribution<int> f(0, features - 1);
    std::uniform_real_distribution<double> c(0.0, side);
    std::vector<colocation::SpatialInstance> inst;
    for (int i = 0; i < n; ++i) inst.push_back({f(rng), c(rng), c(rng)});
    std::vector<std::string> names;
    for (int j = 0; j < features; ++j) names.push_back(std::string(1, static_cast<char>('A' + j)));
    return colocation::SpatialDataset(inst, names);
}

inline colocation::UtilityPosterior randomPosterior(std::mt19937_64& rng, int n, int samples, double meanShift) {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> var(0.01, 1.0);
    colocation::UtilityPosterior p;
    for (int i = 0; i < n; ++i) {
        p.mean.push_back(meanShift + normal(rng));
        p.variance.push_back(var(rng));
    }
    p.bootstrap = colocation::BootstrapMatrix(samples, n);
    for (int b = 0; b < samples; ++b)
        for (int i = 0; i < n; ++i) p.bootstrap.at(b, i) = p.mean[i] + 0.3 * normal(rng);
    return p;
}

// neighbour pairs by checking every pair
inline std::set<std::pair<int, int>> bruteNeighbors(const colocation::SpatialDataset& d, double dist) {
    std::set<std::pair<int, int>> out;
    for (int i = 0; i < d.size(); ++i)
        for (int j = i + 1; j < d.size(); ++j) {
            const auto &a = d.instance(i), &b = d.instance(j);
            const double dx = a.x - b.x, dy = a.y - b.y;
            if (a.feature != b.feature && dx * dx + dy * dy <= dist * dist + 1e-9) out.insert({i, j});
        }
    return out;
}

// row instances of every pattern (size >= 2): all cliques with one instance per feature, by extending cliques one
// instance at a time; returns pattern -> participating instances
inline std::map<std::vector<int>, std::set<int>> brutePatterns(const colocation::SpatialDataset& d, double dist) {
    const auto pairs = bruteNeighbors(d, dist);
    auto adjacent = [&](int a, int b) { return pairs.count({std::min(a, b), std::max(a, b)}) > 0; };
    std::map<std::vector<int>, std::set<int>> out;
    std::vector<std::vector<int>> frontier;
    for (int i = 0; i < d.size(); ++i) frontier.push_back({i});
    while (!frontier.empty()) {
        std::vector<std::vector<int>> next;
        for (const auto& c : frontier)
            for (int j = c.back() + 1; j < d.size(); ++j) {
                bool ok = true;
                for (int i : c) ok = ok && adjacent(i, j);   // adjacency implies different features
                if (!ok) continue;
                std::vector<int> e = c;
                e.push_back(j);
                std::vector<int> pattern;
                for (int i : e) pattern.push_back(d.instance(i).feature);
                std::sort(pattern.begin(), pattern.end());
                if (std::adjacent_find(pattern.begin(), pattern.end()) != pattern.end()) continue;
                for (int i : e) out[pattern].insert(i);
                next.push_back(e);
            }
        frontier.swap(next);
    }
    return out;
}

// Lambda_z of a set of instances, computed directly from the posterior (no use of InstanceUtilities)
inline double bruteScore(const colocation::UtilityPosterior& p, const std::set<int>& a, double z, double normalizer) {
    double m = 0, v = 0;
    for (int i : a) {
        m += p.mean[i];
        v += p.variance[i];
    }
    const int B = p.bootstrap.samples();
    if (B > 1) {
        std::vector<double> sums(B, 0.0);
        for (int b = 0; b < B; ++b)
            for (int i : a) sums[b] += p.bootstrap.at(b, i);
        double avg = 0;
        for (double s : sums) avg += s / B;
        double var = 0;
        for (double s : sums) var += (s - avg) * (s - avg);
        v += var / (B - 1);
    }
    return (m - z * std::sqrt(std::max(v, 0.0))) / normalizer;
}

inline double positiveSum(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += std::max(x, 0.0);
    return s;
}

}  // namespace unit
