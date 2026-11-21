// Yelp businesses, check-ins and reviews -> instance table and rating posteriors (equation (4) of the paper), port of
// uhucm/yelp_full.py (build, vt_boot, expected_pos).
//   business.json : city, coordinates, categories -> one of 20 features (first match in FEATURES)
//   checkin.json  : dated check-ins split at SPLIT / TEST_END (or weekday-hour counts in the old release)
//   reviews_<City>.csv (review_id,user_id,business_id,stars,date): training reviews (date < SPLIT), test reviews
// Posterior per business: theta = (S + kappa_f mu_f) / (n + kappa_f), var sigma_s^2 / (n + kappa_f), with sigma_s^2 the
// pooled within-business variance, mu_f the review-weighted mean and kappa_f = sigma_s^2 / max(tau_f^2, 0.05);
// u = ln(1 + a) (theta - 3) / 2. Total variance: B bootstrap redraws of the mean ratings and within variances,
// hyperparameters estimated again (vt_boot). Normalizer: sum of E[u+] under normal posteriors.
// Usage: yelp_prep BUSINESS_JSON CHECKIN_JSON REVIEW_DIR "City|City|..." SPLIT TEST_END OUT_DIR [--nfeat K] [--B B] [--seed S]
//   -> OUT_DIR/<City>.csv and OUT_DIR/<City>.post (posterior file of io.hpp), JSON summary on stdout
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "io.hpp"
#include "json_lite.hpp"
#include "utm.hpp"

using namespace std;
using namespace uhucpm;

static const vector<pair<string, string>> FEATURES = {
    {"Coffee & Tea", "Coffee"}, {"Fast Food", "FastFood"}, {"Pizza", "Pizza"}, {"Mexican", "Mexican"}, {"Chinese", "Chinese"},
    {"Japanese", "Japanese"}, {"Italian", "Italian"}, {"Burgers", "Burgers"}, {"Breakfast & Brunch", "Breakfast"},
    {"Desserts", "Desserts"}, {"Bars", "Bars"}, {"Hotels", "Hotels"}, {"Nail Salons", "Nails"}, {"Hair Salons", "Hair"},
    {"Fitness & Instruction", "Fitness"}, {"Auto Repair", "AutoRepair"}, {"Dentists", "Dentists"}, {"Grocery", "Grocery"},
    {"Banks & Credit Unions", "Banks"}, {"Fashion", "Fashion"}};
static const map<string, int> UTMZ = {{"Las Vegas", 11}, {"Phoenix", 12}, {"Toronto", 17}, {"Charlotte", 17}, {"Pittsburgh", 17},
                                      {"Calgary", 11}, {"Scottsdale", 12}, {"Mesa", 12}};

static string strip(const string& s) {   // Python str.strip()
    size_t a = 0, b = s.size();
    while (a < b && isspace((unsigned char)s[a])) ++a;
    while (b > a && isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

static string feature_of(const string& cats) {
    set<string> s;
    stringstream ss(cats);
    string c;
    while (getline(ss, c, ',')) s.insert(strip(c));
    if (!cats.empty() && cats.back() == ',') s.insert("");
    for (auto& kv : FEATURES) if (s.count(kv.first)) return kv.second;
    return "";
}

struct Biz { string city, feat; double lat, lon; };

int main(int argc, char** argv) {
    if (argc < 8) { cerr << "usage: yelp_prep BUSINESS CHECKIN REVIEW_DIR CITIES SPLIT TEST_END OUT_DIR [--nfeat K] [--B B] [--seed S]\n"; return 2; }
    const string split = argv[5], test_end = argv[6], outdir = argv[7];
    int nfeat = 0, B = 20;
    unsigned long long seed = 0;
    for (int i = 8; i + 1 < argc; ++i) {
        string a = argv[i];
        if (a == "--nfeat") nfeat = atoi(argv[++i]);
        else if (a == "--B") B = atoi(argv[++i]);
        else if (a == "--seed") seed = strtoull(argv[++i], nullptr, 10);
    }
    vector<string> cities;
    { stringstream ss(argv[4]); string c; while (getline(ss, c, '|')) cities.push_back(c); }
    const set<string> cityset(cities.begin(), cities.end());

    // businesses (insertion order kept)
    vector<string> bid;
    unordered_map<string, Biz> biz;
    {
        ifstream f(argv[1]);
        string line;
        while (getline(f, line)) {
            if (line.empty()) continue;
            auto o = JsonLine(line).parse_object();
            auto ci = o.find("city");
            if (ci == o.end() || ci->second.kind != JVal::Str || !cityset.count(ci->second.str)) continue;
            const string id = o["business_id"].str;
            if (biz.count(id)) continue;
            auto la = o.find("latitude");
            if (la == o.end() || la->second.kind != JVal::Num) continue;
            auto ca = o.find("categories");
            string ft = feature_of(ca != o.end() && ca->second.kind == JVal::Str ? ca->second.str : "");
            if (ft.empty()) continue;
            biz[id] = {ci->second.str, ft, la->second.num, o["longitude"].num};
            bid.push_back(id);
        }
    }
    // check-ins
    unordered_map<string, long long> ck, ck_te;
    bool dated = false;
    {
        ifstream f(argv[2]);
        string line;
        while (getline(f, line)) {
            if (line.empty()) continue;
            auto o = JsonLine(line).parse_object();
            const string id = o["business_id"].str;
            if (!biz.count(id)) continue;
            auto d = o.find("date");
            if (d != o.end()) {
                dated = true;
                stringstream ss(d->second.str);
                string t;
                while (getline(ss, t, ',')) {
                    string day = strip(t).substr(0, 10);
                    if (day < split) ck[id]++;
                    else if (day < test_end) ck_te[id]++;
                }
            } else {
                auto tm = o.find("time");
                if (tm != o.end() && tm->second.kind == JVal::Raw) {
                    string raw = tm->second.str;
                    auto inner = JsonLine(raw).parse_object();
                    long long s = 0;
                    for (auto& kv : inner) s += (long long)kv.second.num;
                    ck[id] += s;
                }
            }
        }
    }
    // reviews
    unordered_map<string, vector<double>> tr, te;
    for (auto& city : cities) {
        string cname = city;
        for (char& c : cname) if (c == ' ') c = '_';
        Table t = read_csv(string(argv[3]) + "/reviews_" + cname + ".csv");
        const int cb = t.col("business_id"), cs = t.col("stars"), cd = t.col("date");
        for (auto& r : t.rows) {
            if (!biz.count(r[cb])) continue;
            const string day = r[cd].substr(0, 10);
            if (day < split) tr[r[cb]].push_back(stod(r[cs]));
            else if (day < test_end) te[r[cb]].push_back(stod(r[cs]));
        }
    }
    cout << setprecision(17) << "{\"dated_checkins\": " << (dated ? "true" : "false") << ", \"cities\": [";
    bool firstc = true;
    for (auto& city : cities) {
        vector<string> ids;
        for (auto& b : bid) if (biz[b].city == city) ids.push_back(b);
        if (ids.empty()) continue;
        vector<double> lons;
        for (auto& b : ids) lons.push_back(biz[b].lon);
        sort(lons.begin(), lons.end());
        int zone = (int)std::floor((lons[lons.size() / 2] + 180) / 6) + 1;
        if (UTMZ.count(city)) zone = UTMZ.at(city);
        // pooled within-business variance
        double num = 0, den = 0;
        for (auto& b : ids) {
            auto& v = tr[b];
            if (v.size() < 2) continue;
            double s = 0;
            for (double x : v) s += x;
            const double m = s / v.size();
            for (double x : v) num += (x - m) * (x - m);
            den += (double)v.size() - 1;
        }
        const double s2s = num / den;
        // hyperparameters per feature (businesses with training reviews)
        map<string, vector<string>> byf;
        for (auto& b : ids) if (!tr[b].empty()) byf[biz[b].feat].push_back(b);
        struct H { double mu_f, tau2, kappa; };
        map<string, H> hyper;
        for (auto& kv : byf) {
            double sa = 0, na = 0, inv = 0;
            vector<double> means;
            for (auto& b : kv.second) {
                double s = 0;
                for (double x : tr[b]) s += x;
                sa += s;
                na += tr[b].size();
                means.push_back(s / tr[b].size());
                inv += 1.0 / tr[b].size();
            }
            double mb = 0;
            for (double x : means) mb += x;
            mb /= means.size();
            double v = 0;
            for (double x : means) v += (x - mb) * (x - mb);
            v /= max<double>(1, (double)means.size() - 1);
            const double tau2 = max(0.05, v - s2s * inv / kv.second.size());
            hyper[kv.first] = {sa / na, tau2, s2s / tau2};
        }
        struct R { string id, feat; double x, y, a, a_test, n, S, SS, nt, mu, s2, mu_test; };
        vector<R> rows;
        for (auto& b : ids) {
            const Biz& z = biz[b];
            if (!hyper.count(z.feat)) continue;
            const H& h = hyper[z.feat];
            const auto& v = tr[b];
            double S = 0, SS = 0;
            for (double x : v) { S += x; SS += x * x; }
            const double n = (double)v.size();
            const double th = (S + h.kappa * h.mu_f) / (n + h.kappa), vth = s2s / (n + h.kappa);
            const double g = std::log1p((double)ck[b]), gt = dated ? std::log1p((double)ck_te[b]) : g;
            double x, y;
            utm_north(z.lon, z.lat, zone, x, y);
            const auto& w = te[b];
            double st = 0;
            for (double q : w) st += q;
            rows.push_back({b, z.feat, x, y, (double)ck[b], dated ? (double)ck_te[b] : -1.0, n, S, SS, (double)w.size(),
                            g * (th - 3.0) / 2, g * g * vth / 4, w.empty() ? NAN : gt * (st / w.size() - 3.0) / 2});
        }
        // optional: keep the nfeat features with the most businesses (ties by name), as the running-time experiment does
        if (nfeat > 0) {
            map<string, int> cnt;
            for (auto& r : rows) cnt[r.feat]++;
            vector<pair<string, int>> v(cnt.begin(), cnt.end());
            stable_sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second > b.second; });
            set<string> keep;
            for (int i = 0; i < nfeat && i < (int)v.size(); ++i) keep.insert(v[i].first);
            vector<R> r2;
            for (auto& r : rows) if (keep.count(r.feat)) r2.push_back(r);
            rows.swap(r2);
        }
        const int n = (int)rows.size();
        // total variance: bootstrap of the hyperparameters (vt_boot)
        vector<double> nn(n), S(n), SS(n), sbar(n), wv(n), df(n), g(n);
        double sdf = 0, swv = 0;
        for (int i = 0; i < n; ++i) {
            nn[i] = rows[i].n; S[i] = rows[i].S; SS[i] = rows[i].SS;
            sbar[i] = nn[i] > 0 ? S[i] / max(nn[i], 1.0) : 0.0;
            wv[i] = nn[i] >= 2 ? (SS[i] - nn[i] * sbar[i] * sbar[i]) / max(nn[i] - 1, 1.0) : 0.0;
            df[i] = max(nn[i] - 1, 0.0);
            sdf += df[i];
            swv += df[i] * wv[i];
            g[i] = std::log1p(rows[i].a);
        }
        const double s2b0 = swv / sdf;
        set<string> fs;
        for (auto& r : rows) fs.insert(r.feat);
        vector<string> feats(fs.begin(), fs.end());
        map<string, int> fidx;
        for (size_t j = 0; j < feats.size(); ++j) fidx[feats[j]] = (int)j;
        vector<int> F(n);
        for (int i = 0; i < n; ++i) F[i] = fidx[rows[i].feat];
        mt19937_64 rng(seed);
        normal_distribution<double> N01(0.0, 1.0);
        vector<double> U((size_t)B * n);
        for (int b = 0; b < B; ++b) {
            vector<double> sb(n), wb(n);
            double s2b_num = 0;
            for (int i = 0; i < n; ++i) {
                sb[i] = nn[i] > 0 ? sbar[i] + std::sqrt(s2b0 / max(nn[i], 1.0)) * N01(rng) : 0.0;
                if (nn[i] >= 2) {
                    chi_squared_distribution<double> C(max(df[i], 1.0));
                    wb[i] = s2b0 * C(rng) / max(df[i], 1.0);
                } else wb[i] = 0.0;
                s2b_num += df[i] * wb[i];
            }
            const double s2b = s2b_num / sdf;
            vector<double> th(n, 0.0);
            for (size_t j = 0; j < feats.size(); ++j) {
                double wn = 0, ws = 0, cntL = 0, inv = 0, sm = 0;
                for (int i = 0; i < n; ++i)
                    if (F[i] == (int)j && nn[i] > 0) { wn += nn[i]; ws += nn[i] * sb[i]; cntL += 1; inv += 1.0 / nn[i]; sm += sb[i]; }
                const double m = cntL > 0 ? ws / wn : 3.0;
                double v = 0;
                if (cntL > 1) {
                    const double mb = sm / cntL;
                    for (int i = 0; i < n; ++i) if (F[i] == (int)j && nn[i] > 0) v += (sb[i] - mb) * (sb[i] - mb);
                    v /= cntL - 1;
                }
                const double tau2 = cntL > 0 ? max(0.05, v - s2b * (inv / cntL)) : 0.05;
                const double kap = s2b / tau2;
                for (int i = 0; i < n; ++i) if (F[i] == (int)j) th[i] = (S[i] + kap * m) / (nn[i] + kap);
            }
            for (int i = 0; i < n; ++i) U[(size_t)b * n + i] = g[i] * (th[i] - 3.0) / 2;
        }
        // expected normalizer
        double Z = 0;
        for (auto& r : rows) {
            const double sd = std::sqrt(max(r.s2, 1e-300)), zz = r.mu / sd;
            Z += r.mu * 0.5 * std::erfc(-zz / std::sqrt(2.0)) + sd * std::exp(-0.5 * zz * zz) / std::sqrt(2 * M_PI);
        }
        string cname = city;
        for (char& c : cname) if (c == ' ') c = '_';
        const string AL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        ofstream o(outdir + "/" + cname + ".csv");
        o << setprecision(17) << "Feature,Instance,LocX,LocY,a,a_test,n_r,S,SS,n_test,mu,s2,mu_test,business_id,feat\n";
        map<int, int> cnt;
        for (int i = 0; i < n; ++i) {
            auto& r = rows[i];
            o << AL[F[i]] << "," << ++cnt[F[i]] << "," << r.x << "," << r.y << "," << r.a << "," << r.a_test << "," << r.n << "," << r.S
              << "," << r.SS << "," << r.nt << "," << r.mu << "," << r.s2 << "," << r.mu_test << "," << r.id << "," << r.feat << "\n";
        }
        PosteriorFile pf;
        for (auto& r : rows) { pf.mu.push_back(r.mu); pf.s2.push_back(r.s2); }
        pf.B = B;
        pf.U = U;
        pf.Z = Z;
        write_posterior(outdir + "/" + cname + ".post", pf);
        cout << (firstc ? "" : ", ") << "{\"city\": \"" << city << "\", \"n\": " << n << ", \"features\": " << feats.size()
             << ", \"sigma_s2\": " << s2s << ", \"zone\": " << zone << ", \"Z\": " << Z << "}";
        firstc = false;
    }
    cout << "]}" << endl;
    return 0;
}
