// Compound empirical Bayes posteriors for count evidence (Definition 4 of the paper), port of uhucm/ceb.py.
// Per feature: x_o ~ Poisson(lambda_o E), prior of theta = ln(lambda) = NPMLE on a grid of J points (EM, `iters`
// iterations), smoothed by a Gaussian kernel of bandwidth 0.9 sd n^(-1/5); utility u = theta - c_f with the moment
// center; instances with a likelihood narrower than 5 grid steps use a local grid of 201 points; t > 0: counts observed
// only when >= t. Outputs posterior mean and variance of u and E[u+ | data] (expected normalizer).
#pragma once
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace uhucpm {

struct CebFeatureState {           // what the bootstrap needs for one feature
    std::vector<int> idx;          // instances of the feature
    std::vector<double> th;        // grid
    std::vector<double> L;         // likelihood of the observed counts on the grid (n_f x J, row major)
    std::vector<double> w;         // unsmoothed NPMLE weights
    double c = 0;                  // center
    int t = 0;
};

struct CebResult {
    std::vector<double> mu, s2, ep;
    std::vector<CebFeatureState> state;
};

double moment_center(const std::vector<double>& x, double E);
double plugin_center(const std::vector<double>& th, const std::vector<double>& w);

// counts, exposure E, feature of each instance (any integers; processed in increasing order)
CebResult ceb_posterior(const std::vector<double>& counts, double E, const std::vector<int>& F, int J = 300,
                        int iters = 300, int t = 0);

// conditional bootstrap matrix U (B x n, row major): counts redrawn x* ~ Poisson(x) with the instances kept, prior and
// center re-estimated (EM from the full-data weights, `iters` iterations), posterior means of the observed counts
std::vector<double> cboot(const std::vector<double>& counts, double E, const CebResult& st, int B = 20, int iters = 40,
                          uint64_t seed = 0);

}  // namespace uhucpm
