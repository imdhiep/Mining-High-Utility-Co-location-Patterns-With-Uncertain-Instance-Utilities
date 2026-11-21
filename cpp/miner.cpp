// UHUCPM miner, port of uhucm/fastmine.py (see miner.hpp).
#include "miner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace uhucpm {

Problem::Problem(const std::vector<Instance>& inst, int k_, double d_) : n((int)inst.size()), k(k_), d(d_) {
    feat.resize(n);
    for (int i = 0; i < n; ++i) feat[i] = inst[i].feat;
    neighbours(inst);
    maximal_cliques();
    C = (int)cliques.size();
    mem.assign((size_t)C * k, -1);
    for (int c = 0; c < C; ++c)
        for (int o : cliques[c]) mem[(size_t)c * k + feat[o]] = o;
}

// grid of cell size d; a pair is a neighbour pair when the squared distance is at most d^2 + 1e-9 and the features differ
void Problem::neighbours(const std::vector<Instance>& inst) {
    nb.assign(n, {});
    std::unordered_map<long long, std::vector<int>> grid;
    auto key = [](long long gx, long long gy) { return (gx << 32) ^ (gy & 0xffffffffLL); };
    std::vector<long long> cx(n), cy(n);
    for (int i = 0; i < n; ++i) {
        cx[i] = (long long)std::floor(inst[i].x / d);
        cy[i] = (long long)std::floor(inst[i].y / d);
        grid[key(cx[i], cy[i])].push_back(i);
    }
    const double d2 = d * d + 1e-9;
    for (int i = 0; i < n; ++i) {
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy) {
                auto it = grid.find(key(cx[i] + dx, cy[i] + dy));
                if (it == grid.end()) continue;
                for (int j : it->second) {
                    if (j == i || feat[j] == feat[i]) continue;
                    double ex = inst[j].x - inst[i].x, ey = inst[j].y - inst[i].y;
                    if (ex * ex + ey * ey <= d2) nb[i].push_back(j);
                }
            }
        std::sort(nb[i].begin(), nb[i].end());
        nb[i].erase(std::unique(nb[i].begin(), nb[i].end()), nb[i].end());
    }
}

static std::vector<int> inter(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> r;
    r.reserve(std::min(a.size(), b.size()));
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(r));
    return r;
}

// Bron-Kerbosch with pivoting in degeneracy order (Eppstein et al.), maximal cliques of size >= 2
void Problem::maximal_cliques() {
    cliques.clear();
    std::vector<int> deg(n, 0), verts;
    for (int v = 0; v < n; ++v)
        if (!nb[v].empty()) { deg[v] = (int)nb[v].size(); verts.push_back(v); }
    // degeneracy order by a bucket queue
    int maxd = 0;
    for (int v : verts) maxd = std::max(maxd, deg[v]);
    std::vector<std::vector<int>> bucket(maxd + 1);
    std::vector<int> where(n, -1), D(deg);
    std::vector<char> removed(n, 0);
    for (int v : verts) bucket[D[v]].push_back(v);
    std::vector<int> order;
    order.reserve(verts.size());
    std::vector<int> pos_in_bucket(n, -1);
    for (int b = 0; b <= maxd; ++b)
        for (size_t t = 0; t < bucket[b].size(); ++t) pos_in_bucket[bucket[b][t]] = (int)t;
    auto bucket_remove = [&](int v) {
        auto& B = bucket[D[v]];
        int p = pos_in_bucket[v];
        int last = B.back();
        B[p] = last;
        pos_in_bucket[last] = p;
        B.pop_back();
    };
    int cur = 0;
    for (size_t step = 0; step < verts.size(); ++step) {
        cur = 0;
        while (bucket[cur].empty()) ++cur;
        int v = bucket[cur].back();
        bucket_remove(v);
        removed[v] = 1;
        order.push_back(v);
        for (int w : nb[v]) {
            if (removed[w]) continue;
            bucket_remove(w);
            --D[w];
            pos_in_bucket[w] = (int)bucket[D[w]].size();
            bucket[D[w]].push_back(w);
        }
    }
    std::vector<int> pos(n, -1);
    for (size_t i = 0; i < order.size(); ++i) pos[order[i]] = (int)i;
    struct Frame { std::vector<int> R, P, X; };
    for (int v : order) {
        Frame f0;
        f0.R = {v};
        for (int w : nb[v]) (pos[w] > pos[v] ? f0.P : f0.X).push_back(w);
        std::vector<Frame> stack;
        stack.push_back(std::move(f0));
        while (!stack.empty()) {
            Frame fr = std::move(stack.back());
            stack.pop_back();
            if (fr.P.empty() && fr.X.empty()) {
                if (fr.R.size() >= 2) {
                    std::vector<int> Q = fr.R;
                    std::sort(Q.begin(), Q.end());
                    cliques.push_back(std::move(Q));
                }
                continue;
            }
            // pivot: vertex of P u X with the most neighbours in P
            int piv = -1;
            size_t best = 0;
            bool first = true;
            for (const auto* S : {&fr.P, &fr.X})
                for (int u : *S) {
                    size_t c = inter(nb[u], fr.P).size();
                    if (first || c > best) { best = c; piv = u; first = false; }
                }
            std::vector<int> cand;
            std::set_difference(fr.P.begin(), fr.P.end(), nb[piv].begin(), nb[piv].end(), std::back_inserter(cand));
            std::vector<int> P = fr.P, X = fr.X;
            for (int u : cand) {
                Frame ch;
                ch.R = fr.R;
                ch.R.push_back(u);
                ch.P = inter(P, nb[u]);
                ch.X = inter(X, nb[u]);
                stack.push_back(std::move(ch));
                P.erase(std::lower_bound(P.begin(), P.end(), u));
                X.insert(std::lower_bound(X.begin(), X.end(), u), u);
            }
        }
    }
    std::sort(cliques.begin(), cliques.end());
}

void Problem::set_values(const std::vector<double>& mu_, const std::vector<double>& s2_,
                         const std::vector<double>& U, int B_, double Z_) {
    if ((int)mu_.size() != n || (int)s2_.size() != n) throw std::runtime_error("set_values: size mismatch");
    mu = mu_;
    s2 = s2_;
    B = B_;
    ub.clear();
    if (B > 0) {
        if ((int)U.size() != B * n) throw std::runtime_error("set_values: bootstrap matrix size");
        ub.assign((size_t)B * n, 0.0);
        const double sc = 1.0 / std::sqrt((double)std::max(B - 1, 1));
        for (int i = 0; i < n; ++i) {
            double m = 0;
            for (int b = 0; b < B; ++b) m += U[(size_t)b * n + i];
            m /= B;
            for (int b = 0; b < B; ++b) ub[(size_t)b * n + i] = (U[(size_t)b * n + i] - m) * sc;
        }
    }
    if (Z_ > 0) Z = Z_;
    else {
        Z = 0;
        for (double m : mu) if (m > 0) Z += m;
    }
    // positive-mean instances ordered by s2/mu with an exact cross-multiplication comparator (stable: ties by index)
    std::vector<int> posi;
    for (int i = 0; i < n; ++i) if (mu[i] > 0) posi.push_back(i);
    std::stable_sort(posi.begin(), posi.end(), [&](int a, int b) { return s2[a] * mu[b] < s2[b] * mu[a]; });
    rank.assign(n, -1);
    smu.resize(posi.size());
    ss2.resize(posi.size());
    for (size_t r = 0; r < posi.size(); ++r) { rank[posi[r]] = (int)r; smu[r] = mu[posi[r]]; ss2[r] = s2[posi[r]]; }
}

double Problem::score_of(const std::vector<int>& A, double z) const {
    double m = 0, v = 0;
    for (int a : A) { m += mu[a]; v += s2[a]; }
    for (int b = 0; b < B; ++b) {
        double s = 0;
        const double* row = &ub[(size_t)b * n];
        for (int a : A) s += row[a];
        v += s * s;
    }
    return (m - z * std::sqrt(std::max(v, 0.0))) / Z;
}

std::vector<int> Problem::pattern_instances(const std::vector<int>& P) const {
    std::vector<char> mark(n, 0);
    for (int c = 0; c < C; ++c) {
        bool all = true;
        for (int f : P) if (mem[(size_t)c * k + f] < 0) { all = false; break; }
        if (!all) continue;
        for (int f : P) mark[mem[(size_t)c * k + f]] = 1;
    }
    std::vector<int> A;
    for (int i = 0; i < n; ++i) if (mark[i]) A.push_back(i);
    return A;
}

namespace {

// SR_z over a set of positive-mean instances given in rank order (prefix property): max(0, max_prefix(M - z sqrt(V)))
double sr_sorted(const std::vector<double>& m, const std::vector<double>& v, double z) {
    double M = 0, V = 0, best = 0;
    for (size_t i = 0; i < m.size(); ++i) {
        M += m[i];
        V += v[i];
        best = std::max(best, M - z * std::sqrt(V));
    }
    return best;
}

}  // namespace

std::map<std::vector<int>, double> Problem::mine(const MineOptions& o, MineStats& st) const {
    using clk = std::chrono::steady_clock;
    const auto t_start = clk::now();
    const double tol = 1e-12, gam = o.gamma, z = o.z;
    const bool none = o.bound == "none";
    const bool mean_only = o.bound == "mean" || z <= 0;
    const bool use_chord = o.bound == "two";
    const int npos = (int)smu.size();
    std::map<std::vector<int>, double> res;

    auto hasf = [&](int c, int f) { return mem[(size_t)c * k + f] >= 0; };
    std::vector<std::vector<int>> Lf(k);
    for (int c = 0; c < C; ++c)
        for (int f = 0; f < k; ++f)
            if (hasf(c, f)) Lf[f].push_back(c);
    std::vector<char> alive(k, 0);
    for (int f = 0; f < k; ++f) alive[f] = !Lf[f].empty();

    // scratch arrays
    std::vector<int> stamp(n, 0);
    int stamp_id = 0;
    std::vector<int> best_key(std::max(npos, 1), -1);

    // bound of a set of instances (rank ids of positive-mean instances), kind as in fastmine.bound
    auto bound_of_ranks = [&](std::vector<int>& r) -> double {
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
        std::vector<double> m, v;
        m.reserve(r.size());
        v.reserve(r.size());
        for (int x : r) { m.push_back(smu[x]); v.push_back(ss2[x]); }
        if (mean_only) { double s = 0; for (double x : m) s += x; return s; }
        return sr_sorted(m, v, z);
    };

    // feature bounds, recomputed until no feature is removed (steps 3-6 of Algorithm 1)
    std::vector<double> fub(k, 0.0);
    while (true) {
        for (int f = 0; f < k; ++f) {
            if (!alive[f]) continue;
            std::vector<int> r;
            for (int c : Lf[f])
                for (int g = 0; g < k; ++g)
                    if (alive[g]) {
                        int a = mem[(size_t)c * k + g];
                        if (a >= 0 && rank[a] >= 0) r.push_back(rank[a]);
                    }
            fub[f] = bound_of_ranks(r) / Z;
            ++st.bound_calls;
        }
        if (none) break;
        bool removed = false;
        std::vector<int> bad;
        for (int f = 0; f < k; ++f)
            if (alive[f] && fub[f] < gam - tol) bad.push_back(f);
        for (int f : bad) { alive[f] = 0; removed = true; }
        if (!removed) break;
    }
    // features whose instances (in cliques) all have a non-positive mean
    std::vector<char> npf(k, 0);
    for (int f = 0; f < k; ++f) {
        if (!alive[f]) continue;
        bool ok = true;
        for (int c : Lf[f]) if (mu[mem[(size_t)c * k + f]] > 0) { ok = false; break; }
        npf[f] = ok;
    }
    std::vector<int> order;
    for (int tier = 0; tier < 2; ++tier) {
        std::vector<int> t;
        for (int f = 0; f < k; ++f) if (alive[f] && (bool)npf[f] == (tier == 1)) t.push_back(f);
        std::sort(t.begin(), t.end(), [&](int a, int b) { return fub[a] != fub[b] ? fub[a] > fub[b] : a < b; });
        order.insert(order.end(), t.begin(), t.end());
    }
    std::vector<int> rk(k, -1);
    for (size_t i = 0; i < order.size(); ++i) rk[order[i]] = (int)i;

    // suffix search at a node: returns true when the bound of the whole subtree prunes; brk = first pruned child position
    auto suffix_search = [&](const std::vector<int>& Lp, const std::vector<int>& P, const std::vector<int>& T,
                             double thr, int& brk) -> bool {
        const int nT = (int)T.size();
        int rmin = npos, rmax = -1;
        for (int c : Lp) {
            int maxpos = -1;
            for (int j = 0; j < nT; ++j) if (hasf(c, T[j])) maxpos = j;
            auto put = [&](int a, int key) {
                int r = rank[a];
                if (r < 0 || key < 0) return;
                if (best_key[r] < key) best_key[r] = key;
                rmin = std::min(rmin, r);
                rmax = std::max(rmax, r);
            };
            for (int f : P) put(mem[(size_t)c * k + f], maxpos);
            for (int j = 0; j < nT; ++j) {
                int a = mem[(size_t)c * k + T[j]];
                if (a >= 0) put(a, j);
            }
        }
        // instances in rank order (the marker array is indexed by rank, so no sorting is needed)
        std::vector<double> um, us;
        std::vector<int> ub_key;
        for (int r = rmin; r <= rmax; ++r)
            if (best_key[r] >= 0) { um.push_back(smu[r]); us.push_back(ss2[r]); ub_key.push_back(best_key[r]); best_key[r] = -1; }
        if (um.empty()) { brk = (0.0 < thr) ? 0 : nT; return 0.0 < thr; }
        std::vector<double> mj, sj;
        auto below = [&](int j) -> bool {
            mj.clear();
            sj.clear();
            double msum = 0;
            for (size_t i = 0; i < um.size(); ++i)
                if (ub_key[i] >= j) { mj.push_back(um[i]); sj.push_back(us[i]); msum += um[i]; }
            if (mj.empty()) return 0.0 < thr;
            if (msum < thr) return true;               // the mean bound already prunes
            if (mean_only) return false;
            if (use_chord) {
                double Vb = 0;
                for (double v : sj) Vb += v;
                double ch = msum;
                if (Vb > 0) {
                    ch = 0;
                    const double sq = std::sqrt(Vb);
                    for (size_t i = 0; i < mj.size(); ++i) ch += std::max(mj[i] - z * sj[i] / sq, 0.0);
                }
                if (ch < thr) return true;
            }
            return sr_sorted(mj, sj, z) < thr;
        };
        ++st.probes;
        if (below(0)) { brk = 0; return true; }
        int lo = 1, hi = nT;                            // below(lo - 1) is false; the answer is in [lo, hi]
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            ++st.probes;
            if (below(mid)) hi = mid;
            else lo = mid + 1;
        }
        brk = lo;
        return false;
    };

    std::vector<char> gneg(k, 0);
    if (o.rule_l == "global") gneg = npf;

    std::function<void(std::vector<int>&, const std::vector<int>&)> visit = [&](std::vector<int>& P,
                                                                                const std::vector<int>& L) {
        ++st.visited;
        if (o.deadline > 0 && st.visited % 64 == 0 &&
            std::chrono::duration<double>(clk::now() - t_start).count() > o.deadline) {
            st.timeout = true;
            throw std::runtime_error("timeout");
        }
        bool has_v = false;
        double v = 0;
        if (P.size() >= 2) {
            ++stamp_id;
            std::vector<int> A;
            for (int c : L)
                for (int f : P) {
                    int a = mem[(size_t)c * k + f];
                    if (stamp[a] != stamp_id) { stamp[a] = stamp_id; A.push_back(a); }
                }
            v = score_of(A, z);
            has_v = true;
            ++st.evaluated;
            if (v >= gam - tol) {
                std::vector<int> key = P;
                std::sort(key.begin(), key.end());
                res[key] = v;
            }
        }
        std::vector<int> T(order.begin() + rk[P.back()] + 1, order.end());
        if (T.empty()) return;
        if (!none && gam > tol) {
            bool alln = true;
            for (int f : P) if (!npf[f]) { alln = false; break; }
            if (alln) for (int f : T) if (!npf[f]) { alln = false; break; }
            if (alln) { ++st.ps4; return; }
        }
        std::vector<int> Lp;
        for (int c : L) {
            for (int g : T) if (hasf(c, g)) { Lp.push_back(c); break; }
        }
        if (Lp.empty()) return;
        const double thr = (gam - tol) * Z;
        int brk = (int)T.size();
        if (!none) {
            int b = 0;
            bool pcb = suffix_search(Lp, P, T, thr, b);
            ++st.bound_calls;
            if (pcb) { ++st.pcb; return; }
            if (b < (int)T.size()) { brk = b; ++st.sb; }
        }
        for (int gi = 0; gi < brk; ++gi) {
            int g = T[gi];
            std::vector<int> Lq;
            for (int c : Lp) if (hasf(c, g)) Lq.push_back(c);
            if (Lq.empty()) continue;
            if (!o.rule_l.empty() && has_v && v < gam - tol) {
                if (o.rule_l == "global" && gneg[g]) { ++st.rule_l; continue; }
                if (o.rule_l == "sum") {
                    ++stamp_id;
                    double m = 0;
                    for (int c : Lq) {
                        int a = mem[(size_t)c * k + g];
                        if (stamp[a] != stamp_id) { stamp[a] = stamp_id; m += mu[a]; }
                    }
                    if (m <= 0) { ++st.rule_l; continue; }
                }
            }
            P.push_back(g);
            visit(P, Lq);
            P.pop_back();
        }
    };

    try {
        for (int f : order) {
            if (Lf[f].empty()) continue;
            std::vector<int> P{f};
            visit(P, Lf[f]);
        }
    } catch (const std::runtime_error&) {
        if (!st.timeout) throw;
    }
    return res;
}

}  // namespace uhucpm
