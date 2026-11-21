// Foursquare TSMC2014 check-ins -> instance table, port of the loading part of uhucm/fsq_loader.py.
// Columns of the input: user, venue, category id, category name, latitude, longitude, time-zone offset (minutes),
// UTC time ("Tue Apr 03 18:00:09 +0000 2012"). Local time = UTC time + offset. Generic categories are excluded, the
// features are the NFEAT categories with the most venues (ties: name order), each venue takes its most frequent
// category (ties: first in the file), training period = local time before 2012-12-01, test period = the rest.
// Instances: venues with at least one training check-in (sorted by venue id), located at the median latitude and
// longitude of their check-ins, projected to UTM. Exposures: whole days of each period / 7 (weeks).
// Usage: fsq_prep TSV UTM_ZONE NFEAT OUT_CSV  -> OUT_CSV (Feature,Instance,LocX,LocY,a,a_test,venue,category), JSON on stdout
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "utm.hpp"

using namespace std;

static const set<string> EXCLUDE = {"Home (private)", "Residential Building (Apartment / Condo)", "Building", "Road", "Bridge",
                                    "Office", "Government Building", "College Academic Building", "Other Great Outdoors",
                                    "Miscellaneous Shop", "Restaurant", "General Entertainment"};

static long long days_from_civil(long long y, unsigned m, unsigned d) {   // H. Hinnant
    y -= m <= 2;
    const long long era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = (unsigned)(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + (long long)doe - 719468;
}

static long long parse_utc(const string& s) {   // "Tue Apr 03 18:00:09 +0000 2012" -> seconds
    static const char* M[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    char wd[4], mon[4], tz[6];
    int d, H, Mi, S, Y;
    if (sscanf(s.c_str(), "%3s %3s %d %d:%d:%d %5s %d", wd, mon, &d, &H, &Mi, &S, tz, &Y) != 8) throw runtime_error("time " + s);
    int m = 0;
    while (m < 12 && strcmp(M[m], mon)) ++m;
    return days_from_civil(Y, m + 1, d) * 86400LL + H * 3600LL + Mi * 60LL + S;
}

static double median(vector<double> v) {
    sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static string latin1_to_utf8(const string& s) {   // the file is read as latin-1, as in fsq_loader
    string o;
    for (unsigned char c : s) {
        if (c < 0x80) o += (char)c;
        else { o += (char)(0xC0 | (c >> 6)); o += (char)(0x80 | (c & 0x3F)); }
    }
    return o;
}

static string clean(string s) {   // feature name as in fsq_loader: " " and "/" -> "_", "&" -> "and"
    string o;
    for (char c : s) {
        if (c == ' ' || c == '/') o += '_';
        else if (c == '&') o += "and";
        else o += c;
    }
    return o;
}

struct Row { string v, cn; double lat, lon; long long ts; };

int main(int argc, char** argv) {
    if (argc < 5) { cerr << "usage: fsq_prep TSV UTM_ZONE NFEAT OUT_CSV\n"; return 2; }
    const int zone = atoi(argv[2]), nfeat = atoi(argv[3]);
    ifstream f(argv[1], ios::binary);
    if (!f) { cerr << "cannot open " << argv[1] << "\n"; return 2; }
    vector<Row> rows;
    string line;
    while (getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        vector<string> c;
        size_t p = 0;
        while (true) {
            size_t q = line.find('\t', p);
            c.push_back(line.substr(p, q == string::npos ? string::npos : q - p));
            if (q == string::npos) break;
            p = q + 1;
        }
        if (c.size() < 8) continue;
        if (EXCLUDE.count(c[3])) continue;
        rows.push_back({c[1], latin1_to_utf8(c[3]), stod(c[4]), stod(c[5]), parse_utc(c[7]) + 60LL * stoll(c[6])});
    }
    // features: categories with the most distinct venues
    map<string, set<string>> venues_of;
    for (auto& r : rows) venues_of[r.cn].insert(r.v);
    vector<pair<string, size_t>> cats;
    for (auto& kv : venues_of) cats.push_back({kv.first, kv.second.size()});
    stable_sort(cats.begin(), cats.end(), [](auto& a, auto& b) { return a.second > b.second; });
    set<string> top;
    for (int i = 0; i < nfeat && i < (int)cats.size(); ++i) top.insert(cats[i].first);
    vector<Row> df;
    for (auto& r : rows) if (top.count(r.cn)) df.push_back(r);
    // one category per venue: most frequent label; ties go to the label that appears first in the file (the order in
    // which pandas value_counts lists equal counts)
    map<string, vector<pair<string, int>>> lab;
    for (auto& r : df) {
        auto& L = lab[r.v];
        bool found = false;
        for (auto& c : L) if (c.first == r.cn) { c.second++; found = true; break; }
        if (!found) L.push_back({r.cn, 1});
    }
    map<string, string> vcat;
    for (auto& kv : lab) {
        string best;
        int bn = -1;
        for (auto& c : kv.second) if (c.second > bn) { bn = c.second; best = c.first; }
        vcat[kv.first] = best;
    }
    // periods
    const long long t1 = days_from_civil(2012, 12, 1) * 86400LL;
    long long t0 = df[0].ts, t2 = df[0].ts;
    for (auto& r : df) { t0 = min(t0, r.ts); t2 = max(t2, r.ts); }
    auto floordiv = [](long long a, long long b) { return a >= 0 ? a / b : -((-a + b - 1) / b); };
    const double Wtr = floordiv(t1 - t0, 86400) / 7.0, Wte = floordiv(t2 - t1, 86400) / 7.0;
    map<string, long long> a, ate;
    map<string, vector<double>> lats, lons;
    for (auto& r : df) {
        (r.ts < t1 ? a : ate)[r.v]++;
        lats[r.v].push_back(r.lat);
        lons[r.v].push_back(r.lon);
    }
    // feature letters by sorted cleaned name (as uhucpm_mine expects single sorted names)
    map<string, int> fid;
    for (auto& kv : a) fid[clean(vcat[kv.first])] = 0;
    const string AL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%()*+-./:;<=>?@[]^_`{|}~";
    int k = 0;
    for (auto& kv : fid) kv.second = k++;
    if (k > (int)AL.size()) { cerr << "too many features\n"; return 2; }
    ofstream o(argv[4]);
    o << setprecision(17) << "Feature,Instance,LocX,LocY,a,a_test,venue,category\n";
    map<int, int> cnt;
    for (auto& kv : a) {
        const string& v = kv.first;
        double x, y;
        uhucpm::utm_north(median(lons[v]), median(lats[v]), zone, x, y);
        int fj = fid[clean(vcat[v])];
        o << AL[fj] << "," << ++cnt[fj] << "," << x << "," << y << "," << kv.second << "," << (ate.count(v) ? ate[v] : 0) << ","
          << v << "," << clean(vcat[v]) << "\n";
    }
    cout << setprecision(17) << "{\"n\": " << a.size() << ", \"features\": " << k << ", \"Wtr\": " << Wtr << ", \"Wte\": " << Wte
         << ", \"checkins\": " << df.size() << "}" << endl;
    return 0;
}
