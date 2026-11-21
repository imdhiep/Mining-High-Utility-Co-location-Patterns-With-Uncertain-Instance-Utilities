#include "mining/set_rule.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "stats/special_functions.hpp"

namespace colocation {

SetSelection SetRule::select(const std::vector<double>& M, const std::vector<double>& V, double threshold, double delta) {
    if (M.size() != V.size()) throw std::invalid_argument("means and variances differ in size");
    const std::size_t n = M.size();
    SetSelection out;
    out.selected.assign(n, false);
    out.probability.resize(n);
    for (std::size_t i = 0; i < n; ++i) out.probability[i] = SpecialFunctions::normalCdf((M[i] - threshold) / std::sqrt(V[i]));
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return out.probability[a] > out.probability[b]; });
    std::size_t keep = 0;
    double missed = 0;
    for (std::size_t r = 0; r < n; ++r) {
        missed += 1 - out.probability[order[r]];
        if (missed / static_cast<double>(r + 1) <= delta) keep = r + 1;   // the largest such prefix, not the first failure
    }
    for (std::size_t r = 0; r < keep; ++r) out.selected[order[r]] = true;
    return out;
}

}  // namespace colocation
