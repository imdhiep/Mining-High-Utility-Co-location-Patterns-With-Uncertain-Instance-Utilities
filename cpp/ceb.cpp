// Port of uhucm/ceb.py (see ceb.hpp). Array operations follow the numpy code one by one.
#include "ceb.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>

namespace uhucpm {

namespace {

const int LOCAL_PTS = 201;
const double LOCAL_SD = 7.0, LOCAL_STEPS = 5.0;

std::vector<double> linspace(double lo, double hi, int J) {   // numpy.linspace (endpoint included)
    std::vector<double> v(J);
    const double step = (hi - lo) / (J - 1);
    for (int i = 0; i < J; ++i) v[i] = lo + i * step;
    v[J - 1] = hi;
    return v;
}

std::vector<double> grid(const std::vector<double>& x, double E, int J) {
    double mn = *std::min_element(x.begin(), x.end()), mx = *std::max_element(x.begin(), x.end());
    return linspace(std::log((mn + 0.5) / E) - 3, std::log((mx + 0.5) / E) + 1, J);
}

// log P(X >= t) for X ~ Poisson(lam)
double log_sf(int t, double lam) {
    if (t <= 0) return 0.0;
    double p;
    if (t == 1) p = -std::expm1(-lam);
    else {
        double term = std::exp(-lam), cdf = term;
        for (int i = 1; i < t; ++i) { term *= lam / i; cdf += term; }
        p = 1.0 - cdf;
    }
    return std::log(std::max(p, 1e-300));
}

// Poisson likelihood on the grid (n x J), each row divided by its maximum
std::vector<double> lik(const std::vector<double>& x, const std::vector<double>& th, double E, int t) {
    const int n = (int)x.size(), J = (int)th.size();
    std::vector<double> eth(J), lsf(J, 0.0), L((size_t)n * J);
    for (int j = 0; j < J; ++j) { eth[j] = E * std::exp(th[j]); if (t > 0) lsf[j] = log_sf(t, eth[j]); }
    for (int i = 0; i < n; ++i) {
        double* row = &L[(size_t)i * J];
        const double lg = std::lgamma(x[i] + 1);
        double mx = -INFINITY;
        for (int j = 0; j < J; ++j) {
            row[j] = x[i] * th[j] - eth[j] - lg - lsf[j];
            mx = std::max(mx, row[j]);
        }
        for (int j = 0; j < J; ++j) row[j] = std::exp(row[j] - mx);
    }
    return L;
}

std::vector<double> em(const std::vector<double>& L, int n, int J, std::vector<double> w, int iters) {
    std::vector<double> acc(J), post(J);
    for (int it = 0; it < iters; ++it) {
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int i = 0; i < n; ++i) {
            const double* row = &L[(size_t)i * J];
            double s = 0;
            for (int j = 0; j < J; ++j) { post[j] = row[j] * w[j]; s += post[j]; }
            for (int j = 0; j < J; ++j) acc[j] += post[j] / s;
        }
        for (int j = 0; j < J; ++j) w[j] = std::max(acc[j] / n, 1e-300);
    }
    return w;
}

std::vector<double> smooth_prior(const std::vector<double>& th, const std::vector<double>& w, int n) {
    const int J = (int)th.size();
    double m = 0, m2 = 0;
    for (int j = 0; j < J; ++j) { m += w[j] * th[j]; m2 += w[j] * th[j] * th[j]; }
    const double sd = std::sqrt(std::max(m2 - m * m, 1e-12)), h = 0.9 * sd * std::pow((double)n, -0.2);
    if (h <= 0.5 * (th[1] - th[0])) return w;
    std::vector<double> ws(J, 0.0);
    double tot = 0;
    for (int i = 0; i < J; ++i) {
        double s = 0;
        for (int j = 0; j < J; ++j) { double z = (th[i] - th[j]) / h; s += std::exp(-0.5 * z * z) * w[j]; }
        ws[i] = s;
        tot += s;
    }
    for (double& v : ws) v /= tot;
    return ws;
}

double interp(double x, const std::vector<double>& xp, const std::vector<double>& fp) {   // numpy.interp
    if (x <= xp.front()) return fp.front();
    if (x >= xp.back()) return fp.back();
    size_t k = std::upper_bound(xp.begin(), xp.end(), x) - xp.begin();   // xp[k-1] <= x < xp[k]
    const double t = (x - xp[k - 1]) / (xp[k] - xp[k - 1]);
    return fp[k - 1] + t * (fp[k] - fp[k - 1]);
}

struct Moments {
    std::vector<double> mm, vv, post;          // post: n x J
    std::vector<int> sharp;
    std::vector<std::vector<double>> tl, q;    // local grids and posteriors of the sharp instances
};

Moments post_moments(const std::vector<double>& x, double E, const std::vector<double>& th, const std::vector<double>& w,
                     int t, const std::vector<double>& L, bool want_post) {
    const int n = (int)x.size(), J = (int)th.size();
    Moments r;
    r.mm.resize(n);
    r.vv.resize(n);
    if (want_post) r.post.resize((size_t)n * J);
    std::vector<double> p(J);
    for (int i = 0; i < n; ++i) {
        const double* row = &L[(size_t)i * J];
        double s = 0;
        for (int j = 0; j < J; ++j) { p[j] = row[j] * w[j]; s += p[j]; }
        double m = 0, m2 = 0;
        for (int j = 0; j < J; ++j) { p[j] /= s; m += p[j] * th[j]; m2 += p[j] * th[j] * th[j]; }
        r.mm[i] = m;
        r.vv[i] = std::max(m2 - m * m, 1e-12);
        if (want_post) std::copy(p.begin(), p.end(), r.post.begin() + (size_t)i * J);
    }
    const double h = th[1] - th[0];
    std::vector<double> lg(J);
    for (int j = 0; j < J; ++j) lg[j] = std::log(std::max(w[j], 1e-300));
    const std::vector<double> base = linspace(-LOCAL_SD, LOCAL_SD, LOCAL_PTS);
    for (int i = 0; i < n; ++i) {
        const double sd = 1.0 / std::sqrt(x[i] + 0.5);
        if (!(sd < LOCAL_STEPS * h)) continue;
        r.sharp.push_back(i);
        std::vector<double> tl(LOCAL_PTS), ll(LOCAL_PTS), q(LOCAL_PTS);
        const double c0 = std::log((x[i] + 0.5) / E);
        double mx = -INFINITY;
        for (int k = 0; k < LOCAL_PTS; ++k) {
            tl[k] = c0 + sd * base[k];
            ll[k] = x[i] * tl[k] - E * std::exp(tl[k]) + interp(tl[k], th, lg);
            if (t > 0) ll[k] -= log_sf(t, E * std::exp(tl[k]));
            mx = std::max(mx, ll[k]);
        }
        double s = 0;
        for (int k = 0; k < LOCAL_PTS; ++k) { q[k] = std::exp(ll[k] - mx); s += q[k]; }
        double m = 0, m2 = 0;
        for (int k = 0; k < LOCAL_PTS; ++k) { q[k] /= s; m += q[k] * tl[k]; m2 += q[k] * tl[k] * tl[k]; }
        r.mm[i] = m;
        r.vv[i] = std::max(m2 - m * m, 1e-12);
        if (want_post) { r.tl.push_back(std::move(tl)); r.q.push_back(std::move(q)); }
    }
    return r;
}

}  // namespace

double moment_center(const std::vector<double>& x, double E) {
    const double n = (double)x.size();
    double sx = 0, sxx = 0;
    for (double v : x) { sx += v; sxx += v * (v - 1); }
    double m = sx / n / E;
    const double v = std::max(sxx / n / (E * E) - m * m + m / (n * E), 0.0);
    m = std::max(m, 1e-12);
    return std::log(m) - 0.5 * std::log1p(v / (m * m));
}

double plugin_center(const std::vector<double>& th, const std::vector<double>& w) {
    double a = 0, b = 0;
    for (size_t j = 0; j < th.size(); ++j) { const double l = std::exp(th[j]); a += w[j] * l; b += w[j] * l * l; }
    return 2 * std::log(a) - 0.5 * std::log(b);
}

CebResult ceb_posterior(const std::vector<double>& counts, double E, const std::vector<int>& F, int J, int iters, int t) {
    const int n = (int)counts.size();
    CebResult R;
    R.mu.assign(n, 0.0);
    R.s2.assign(n, 0.0);
    R.ep.assign(n, 0.0);
    std::set<int> feats(F.begin(), F.end());
    for (int f : feats) {
        CebFeatureState S;
        for (int i = 0; i < n; ++i) if (F[i] == f) S.idx.push_back(i);
        std::vector<double> x;
        for (int i : S.idx) x.push_back(counts[i]);
        const int nf = (int)x.size();
        S.th = grid(x, E, J);
        S.L = lik(x, S.th, E, t);
        S.w = em(S.L, nf, J, std::vector<double>(J, 1.0 / J), iters);
        const std::vector<double> ws = smooth_prior(S.th, S.w, nf);
        S.c = t == 0 ? moment_center(x, E) : plugin_center(S.th, S.w);
        S.t = t;
        Moments M = post_moments(x, E, S.th, ws, t, S.L, true);
        for (int a = 0; a < nf; ++a) {
            double e = 0;
            for (int j = 0; j < J; ++j) e += M.post[(size_t)a * J + j] * std::max(S.th[j] - S.c, 0.0);
            R.ep[S.idx[a]] = e;
        }
        for (size_t s = 0; s < M.sharp.size(); ++s) {
            double e = 0;
            for (int k = 0; k < LOCAL_PTS; ++k) e += M.q[s][k] * std::max(M.tl[s][k] - S.c, 0.0);
            R.ep[S.idx[M.sharp[s]]] = e;
        }
        for (int a = 0; a < nf; ++a) { R.mu[S.idx[a]] = M.mm[a] - S.c; R.s2[S.idx[a]] = M.vv[a]; }
        R.state.push_back(std::move(S));
    }
    return R;
}

std::vector<double> cboot(const std::vector<double>& counts, double E, const CebResult& st, int B, int iters, uint64_t seed) {
    const int n = (int)counts.size();
    std::vector<double> U((size_t)B * n, 0.0);
    std::mt19937_64 rng(seed);
    auto pois = [&](double m) -> double {
        if (m <= 0) return 0.0;
        std::poisson_distribution<long long> P(m);
        return (double)P(rng);
    };
    for (const CebFeatureState& S : st.state) {
        std::vector<double> x;
        for (int i : S.idx) x.push_back(counts[i]);
        const int nf = (int)x.size(), J = (int)S.th.size();
        for (int b = 0; b < B; ++b) {
            std::vector<double> xb(nf);
            for (int a = 0; a < nf; ++a) xb[a] = pois(x[a]);
            if (S.t > 0) {
                bool any = true;
                while (any) {
                    any = false;
                    for (int a = 0; a < nf; ++a)
                        if (xb[a] < S.t) { xb[a] = pois(x[a]); any = any || xb[a] < S.t; }
                }
            }
            std::vector<double> Lb = lik(xb, S.th, E, S.t);
            std::vector<double> w = em(Lb, nf, J, S.w, iters);
            Moments M = post_moments(x, E, S.th, smooth_prior(S.th, w, nf), S.t, S.L, false);
            const double c = S.t == 0 ? moment_center(xb, E) : plugin_center(S.th, w);
            for (int a = 0; a < nf; ++a) U[(size_t)b * n + S.idx[a]] = M.mm[a] - c;
        }
    }
    return U;
}

}  // namespace uhucpm
