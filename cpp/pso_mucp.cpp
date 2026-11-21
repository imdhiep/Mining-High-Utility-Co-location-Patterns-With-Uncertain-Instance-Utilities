// PSO-MUCP (Bao et al., Data Min. Knowl. Disc. 40:38, 2026), C++ port of the authors' HUCPM/PSO-MUCP.py
// (github.com/yuanshuaikang/PSO-MUCP). Every step follows the Python code, including its iteration orders:
//   Compute_utility, grid_method (cells of size D anchored at the minimum coordinates, 3 x 3 block), Prune_one (FUUR,
//   features with Up/ALL <= Min_utility removed, positive features by increasing Up, then the others), Enum_Cliques,
//   con_hash_table, Find_High_utility_pattern with gener_k (PUUR, no-hope prefixes, Strategy 4, key existence).
// Python lists searched with `in` are replaced by hash sets; this changes the cost of a membership test, not its result.
// Usage: pso_mucp DATA_CSV UTILITY MIN_UTILITY D [--norm NORM]
//   UTILITY  "A:2,B:-1,..." or a Python dict literal "{'A': 2, 'B': -1}"
//   --norm   the normalizer ALL; default as in the authors' code: sum over features of count x utility (all signs)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <sys/resource.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "io.hpp"

using namespace std;

struct Inst { char f; string id; double x, y; };

static map<char, double> parse_utility(string s) {
    map<char, double> u;
    for (char& c : s) if (c == '{' || c == '}' || c == '\'' || c == '"' || c == ' ') c = ' ';
    string t;
    for (char c : s) if (c != ' ') t += c;
    size_t p = 0;
    while (p < t.size()) {
        size_t q = t.find(',', p);
        if (q == string::npos) q = t.size();
        string item = t.substr(p, q - p);
        size_t c = item.find(':');
        if (c != string::npos && c > 0) u[item[0]] = stod(item.substr(c + 1));
        p = q + 1;
    }
    return u;
}

int main(int argc, char** argv) {
    const auto t0 = chrono::steady_clock::now();
    if (argc < 5) { cerr << "usage: pso_mucp DATA_CSV UTILITY MIN_UTILITY D [--norm NORM]\n"; return 2; }
    const map<char, double> Utility = parse_utility(argv[2]);
    const double Min_utility = atof(argv[3]), D = atof(argv[4]);
    bool has_norm = false;
    double norm = 0;
    for (int i = 5; i < argc; ++i)
        if (!strcmp(argv[i], "--norm") && i + 1 < argc) { has_norm = true; norm = atof(argv[++i]); }

    // ---- read instances (Feature, Instance, LocX, LocY): id = feature + instance number
    uhucpm::Table tab = uhucpm::read_csv(argv[1]);
    vector<Inst> I;
    for (auto& r : tab.rows) I.push_back({r[0][0], r[0] + r[1], stod(r[2]), stod(r[3])});
    const int n = (int)I.size();
    auto U = [&](char f) -> double {
        auto it = Utility.find(f);
        if (it == Utility.end()) { cerr << "no utility for feature " << f << "\n"; exit(2); }
        return it->second;
    };

    // ---- Compute_utility: ALL = sum over features of count * utility
    double ALL = 0;
    for (auto& kv : Utility) {
        int cnt = 0;
        for (auto& o : I) if (o.f == kv.first) ++cnt;
        ALL += cnt * kv.second;
    }
    if (has_norm) ALL = norm;

    // ---- grid_method: Ns in the insertion order of the Python dicts
    double xmin = I[0].x, ymin = I[0].y;
    for (auto& o : I) { xmin = min(xmin, o.x); ymin = min(ymin, o.y); }
    map<pair<long long, long long>, int> cell_id;
    vector<pair<long long, long long>> cells;                 // insertion order
    vector<vector<int>> cell_inst;
    for (int i = 0; i < n; ++i) {
        long long cx = (long long)ceil((I[i].x - xmin) / D), cy = (long long)ceil((I[i].y - ymin) / D);
        if (cx == 0) cx += 1;
        if (cy == 0) cy += 1;
        auto key = make_pair(cx, cy);
        auto it = cell_id.find(key);
        if (it == cell_id.end()) { cell_id[key] = (int)cells.size(); cells.push_back(key); cell_inst.push_back({i}); }
        else cell_inst[it->second].push_back(i);
    }
    vector<int> ns_order;                                      // keys of Ns in insertion order
    vector<vector<int>> Ns(n);
    const double d2 = D * D;
    auto close = [&](int a, int b) { double dx = I[a].x - I[b].x, dy = I[a].y - I[b].y; return dx * dx + dy * dy <= d2; };
    for (size_t c = 0; c < cells.size(); ++c) {
        for (int e : cell_inst[c]) {
            ns_order.push_back(e);
            for (int x : cell_inst[c]) if (I[x].f != I[e].f && close(x, e)) Ns[e].push_back(x);
            for (int dx = -1; dx <= 1; ++dx)
                for (int dy = -1; dy <= 1; ++dy) {
                    if (dx == 0 && dy == 0) continue;
                    auto it = cell_id.find({cells[c].first + dx, cells[c].second + dy});
                    if (it == cell_id.end()) continue;
                    for (int x : cell_inst[it->second]) if (I[x].f != I[e].f && close(x, e)) Ns[e].push_back(x);
                }
        }
    }

    // ---- Prune_one
    vector<char> fseen;
    map<char, unordered_set<int>> Ins_number;
    vector<char> feat_order;                                   // order in which features appear among the keys
    vector<char> positive, negative;
    for (int key : ns_order) {
        char f = I[key].f;
        if (!Ins_number.count(f)) {
            Ins_number[f];
            feat_order.push_back(f);
            if (U(f) > 0) positive.push_back(f); else negative.push_back(f);
        }
        auto& S = Ins_number[f];
        for (int x : Ns[key]) S.insert(x);
        S.insert(key);
    }
    map<char, double> Up;
    for (char f : feat_order) {
        double s = 0;
        for (int x : Ins_number[f]) if (U(I[x].f) > 0) s += U(I[x].f);
        Up[f] = s;
    }
    vector<char> del;
    auto is_in = [](const vector<char>& v, char c) { return find(v.begin(), v.end(), c) != v.end(); };
    vector<char> kept;
    for (char f : feat_order) { if (Up[f] / ALL <= Min_utility) del.push_back(f); else kept.push_back(f); }
    vector<char> pk, nk;
    for (char f : kept) (is_in(positive, f) ? pk : nk).push_back(f);
    stable_sort(pk.begin(), pk.end(), [&](char a, char b) { return Up[a] < Up[b]; });
    stable_sort(nk.begin(), nk.end(), [&](char a, char b) { return Up[a] < Up[b]; });
    vector<char> sort_list = pk;
    sort_list.insert(sort_list.end(), nk.begin(), nk.end());
    // delete_nis: keys and neighbours of deleted features removed
    vector<int> dn_order;
    vector<vector<int>> dnis(n);
    vector<unordered_set<int>> dnis_set(n);
    for (int key : ns_order) {
        if (is_in(del, I[key].f)) continue;
        dn_order.push_back(key);
        for (int x : Ns[key]) if (!is_in(del, I[x].f)) dnis[key].push_back(x);
        dnis_set[key] = unordered_set<int>(dnis[key].begin(), dnis[key].end());
    }

    // ---- Enum_Cliques and con_hash_table (inner dict: feature -> instances, both in insertion order)
    struct Entry { vector<char> feats; map<char, vector<int>> inst; map<char, unordered_set<int>> has; };
    vector<string> hash_keys;
    unordered_map<string, int> hk_index;
    vector<Entry> table;
    long long clique_count = 0;
    auto add_clique = [&](int key, const vector<int>& cl) {
        ++clique_count;
        string hk(1, I[key].f);
        for (int e : cl) hk += I[e].f;
        sort(hk.begin(), hk.end());
        auto it = hk_index.find(hk);
        int id;
        if (it == hk_index.end()) { id = (int)table.size(); hk_index[hk] = id; hash_keys.push_back(hk); table.emplace_back(); }
        else id = it->second;
        Entry& E = table[id];
        auto put = [&](int o) {
            char f = I[o].f;
            if (!E.inst.count(f)) E.feats.push_back(f);
            if (E.has[f].insert(o).second) E.inst[f].push_back(o);
        };
        put(key);
        for (int e : cl) put(e);
    };
    for (int key : dn_order) {
        vector<int> flag_list = dnis[key];
        size_t head = 0;
        while (head < flag_list.size()) {
            int current = flag_list[head++];
            const auto& nbc = dnis_set[current];
            vector<int> inter;
            for (size_t t = head; t < flag_list.size(); ++t) if (nbc.count(flag_list[t])) inter.push_back(flag_list[t]);
            vector<pair<vector<int>, vector<int>>> stack;
            stack.push_back({{current}, inter});
            while (!stack.empty()) {
                auto [can_key, in] = std::move(stack.back());
                stack.pop_back();
                if (in.empty()) { add_clique(key, can_key); continue; }
                if (in.size() == 1) { vector<int> c = can_key; c.push_back(in[0]); add_clique(key, c); continue; }
                int first = in[0];
                vector<int> nck = can_key;
                nck.push_back(first);
                vector<int> remaining(in.begin() + 1, in.end());
                vector<int> ni;
                for (int x : remaining) if (dnis_set[first].count(x)) ni.push_back(x);
                stack.push_back({nck, ni});
                stack.push_back({can_key, remaining});
            }
        }
    }

    // ---- Find_High_utility_pattern
    auto pos_in = [&](char c) { for (size_t i = 0; i < sort_list.size(); ++i) if (sort_list[i] == c) return (int)i; return -1; };
    auto order_fn = [&](char e1, const string& e2) {      // order(sort_list, elem_1, elem_2)
        int s1 = -1, s2 = -1;
        for (size_t i = 0; i < sort_list.size(); ++i) {
            if (sort_list[i] == e2.back()) s1 = (int)i;
            if (sort_list[i] == e1) s2 = (int)i;
        }
        return s2 > s1;
    };
    auto contains_all = [](const string& key, const string& cand) {
        for (char c : cand) if (key.find(c) == string::npos) return false;
        return true;
    };
    vector<string> high_utility, no_hope;
    long long all_pattern_num = 0, evaluated = 0;
    vector<string> cand_order;
    map<string, pair<double, double>> Can;              // [0] = PUUR numerator, [1] = utility numerator
    for (char c : sort_list) if (!is_in(negative, c)) { string s(1, c); cand_order.push_back(s); Can[s] = {0, 0}; }
    while (!cand_order.empty()) {
        evaluated += (long long)cand_order.size();
        map<string, map<char, unordered_set<int>>> nh1, up1;
        for (size_t h = 0; h < table.size(); ++h) {
            const string& hk = hash_keys[h];
            const Entry& E = table[h];
            for (const string& ck : cand_order) {
                map<char, vector<int>> number_hash, up_number;
                if (contains_all(hk, ck)) {
                    auto& NH1 = nh1[ck];
                    auto& UP1 = up1[ck];
                    for (char ek : E.feats) {
                        bool in_c = ck.find(ek) != string::npos;
                        bool neg = is_in(negative, ek);
                        if (in_c) {
                            auto& lst = number_hash[ek];
                            auto& seen = NH1[ek];
                            for (int o : E.inst.at(ek)) if (!seen.count(o)) lst.push_back(o);
                        }
                        bool ord = order_fn(ek, ck);
                        if (!up_number.count(ek) && (in_c || ord) && !neg) { up_number[ek]; UP1[ek]; }
                        if ((!in_c && !ord) || neg) continue;
                        auto& seen = UP1[ek];
                        for (int o : E.inst.at(ek)) if (hk != ck && !seen.count(o)) up_number[ek].push_back(o);
                    }
                    for (auto& kv : number_hash) for (int o : kv.second) NH1[kv.first].insert(o);
                    for (auto& kv : up_number) for (int o : kv.second) UP1[kv.first].insert(o);
                }
                for (auto& kv : number_hash) Can[ck].second += (double)kv.second.size() * U(kv.first);
                for (auto& kv : up_number) Can[ck].first += (double)kv.second.size() * U(kv.first);
            }
        }
        for (const string& ck : cand_order) if (Can[ck].second / ALL >= Min_utility) high_utility.push_back(ck);
        // gener_k
        vector<string> next;
        map<string, pair<double, double>> nCan;
        for (const string& k1 : cand_order) {
            if (Can[k1].first / ALL < Min_utility) { no_hope.push_back(k1); continue; }
            for (size_t i = 0; i < sort_list.size(); ++i) {
                if (Can[k1].second / ALL <= Min_utility && is_in(negative, sort_list[i])) continue;
                int x = 0;
                for (size_t j = 0; j < sort_list.size(); ++j) if (k1.back() == sort_list[j]) { x = (int)j; break; }
                if (x >= (int)i) continue;
                if (k1.find(sort_list[i]) != string::npos) continue;
                string kc = k1 + sort_list[i];
                bool hopeless = false;
                for (auto& e : no_hope) if (kc.compare(0, e.size(), e) == 0 && kc.size() >= e.size()) { hopeless = true; break; }
                if (hopeless) continue;
                bool exists = false;
                for (auto& hk : hash_keys) if (contains_all(hk, kc)) { exists = true; break; }
                if (!exists) continue;
                if (!nCan.count(kc)) next.push_back(kc);
                nCan[kc] = {0, 0};
            }
        }
        cand_order = next;
        Can = nCan;
        all_pattern_num += (long long)cand_order.size();
    }
    (void)pos_in;
    vector<string> H;
    for (auto& k : high_utility) if (k.size() > 1) { string s = k; sort(s.begin(), s.end()); H.push_back(s); }
    sort(H.begin(), H.end());
    H.erase(unique(H.begin(), H.end()), H.end());
    const double sec = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    cout << setprecision(17) << "{\"mode\": \"psomucp_cpp\", \"n\": " << H.size() << ", \"patterns\": [";
    for (size_t i = 0; i < H.size(); ++i) cout << (i ? ", " : "") << "\"" << H[i] << "\"";
    cout << "], \"seconds\": " << sec << ", \"cliques\": " << clique_count << ", \"hash_keys\": " << table.size()
         << ", \"evaluated\": " << evaluated << ", \"all_pattern_num\": " << all_pattern_num << ", \"ALL\": " << ALL
         << ", \"max_rss_kb\": " << ru.ru_maxrss << "}" << endl;
    return 0;
}
