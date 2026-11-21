// UHUCPM miner (Section V of the paper): exact mining of the patterns P with
//   Lambda_z(P) = (M(P) - z sqrt(V(P))) / Z >= gamma,
// M(P) = sum of the posterior means of PI(P), V(P) = sum of the posterior variances plus, when a bootstrap matrix is
// given, the variance over the bootstrap samples of the summed posterior means (total variance V_T).
// Port of uhucm/fastmine.py (Problem + mine) restricted to the setting of the paper (no pair atoms, rho = 0).
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace uhucpm {

struct Instance {                 // one spatial instance
    int feat;                     // feature index 0..k-1
    double x, y;
};

struct MineOptions {
    double gamma = 0.0;           // threshold on the normalized score
    double z = 1.645;             // z of the lower bound
    std::string bound = "two";    // "two" (mean, chord, then prefix scan), "exact" (mean, then prefix scan),
                                  // "mean" (mean bound only), "none" (no pruning)
    std::string rule_l;           // "", "global" or "sum": the unsafe local rule of Section III-D (audit only)
    double deadline = 0.0;        // seconds, 0 = none
};

struct MineStats {
    long long visited = 0, evaluated = 0, bound_calls = 0, pcb = 0, sb = 0, ps4 = 0, rule_l = 0, probes = 0;
    bool timeout = false;
};

class Problem {
public:
    Problem(const std::vector<Instance>& inst, int k, double d);
    // posterior means and variances; U: B x n bootstrap matrix of posterior means (row major), may be empty;
    // Z <= 0 means: sum of the positive means
    void set_values(const std::vector<double>& mu, const std::vector<double>& s2,
                    const std::vector<double>& U, int B, double Z);
    // all patterns with score >= gamma; key = sorted feature indices
    std::map<std::vector<int>, double> mine(const MineOptions& o, MineStats& st) const;
    // participating instances of a pattern (sorted feature indices)
    std::vector<int> pattern_instances(const std::vector<int>& P) const;
    double score_of(const std::vector<int>& A, double z) const;

    int n, k;
    double d;
    std::vector<int> feat;
    std::vector<std::vector<int>> nb;          // neighbours (different features, distance <= d)
    std::vector<std::vector<int>> cliques;     // maximal cliques of size >= 2 (sorted instance ids)
    std::vector<int32_t> mem;                  // C x k: instance of feature f in clique c, or -1
    int C = 0;
    std::vector<double> mu, s2, ub;            // ub: B x n, centered and divided by sqrt(B - 1)
    int B = 0;
    double Z = 0.0;
    std::vector<int> rank;                     // rank of positive-mean instances by s2/mu, -1 otherwise
    std::vector<double> smu, ss2;              // means and variances in rank order
private:
    void neighbours(const std::vector<Instance>& inst);
    void maximal_cliques();
};

}  // namespace uhucpm
