// One UHUCPM run: reads the instance file (Feature,Instance,LocX,LocY,...) shared with the other miners and a posterior
// file (io.hpp), mines all patterns with Lambda_z >= gamma and prints one JSON line.
// Usage: uhucpm_mine DATA_CSV POSTERIOR GAMMA D Z [--bound two|exact|mean|none] [--rule global|sum]
//                    [--deadline SEC] [--scores]
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <sys/resource.h>

#include "io.hpp"
#include "miner.hpp"

using namespace uhucpm;

int main(int argc, char** argv) {
    const auto t0 = std::chrono::steady_clock::now();
    if (argc < 6) {
        std::cerr << "usage: uhucpm_mine DATA_CSV POSTERIOR GAMMA D Z [--bound B] [--rule R] [--deadline S] [--scores]\n";
        return 2;
    }
    MineOptions o;
    o.gamma = std::atof(argv[3]);
    double d = std::atof(argv[4]);
    o.z = std::atof(argv[5]);
    bool scores = false;
    for (int i = 6; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--bound") && i + 1 < argc) o.bound = argv[++i];
        else if (!std::strcmp(argv[i], "--rule") && i + 1 < argc) o.rule_l = argv[++i];
        else if (!std::strcmp(argv[i], "--deadline") && i + 1 < argc) o.deadline = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--scores")) scores = true;
        else { std::cerr << "unknown option " << argv[i] << "\n"; return 2; }
    }
    Table t = read_csv(argv[1]);
    const int cf = t.col("Feature"), cx = t.col("LocX"), cy = t.col("LocY");
    std::vector<std::string> names;
    for (auto& r : t.rows) names.push_back(r[cf]);
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    std::map<std::string, int> fi;
    for (size_t j = 0; j < names.size(); ++j) fi[names[j]] = (int)j;
    std::vector<Instance> inst;
    inst.reserve(t.rows.size());
    for (auto& r : t.rows) inst.push_back({fi[r[cf]], std::stod(r[cx]), std::stod(r[cy])});
    PosteriorFile pf = read_posterior(argv[2]);
    if ((size_t)pf.n != inst.size()) { std::cerr << "posterior size " << pf.n << " != " << inst.size() << "\n"; return 2; }

    Problem pb(inst, (int)names.size(), d);
    pb.set_values(pf.mu, pf.s2, pf.U, (int)pf.B, pf.Z);
    MineStats st;
    auto res = pb.mine(o, st);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);

    std::vector<std::pair<std::string, double>> out;
    for (auto& kv : res) {
        std::string s;
        for (int f : kv.first) s += names[f];
        std::vector<std::string> parts;
        for (int f : kv.first) parts.push_back(names[f]);
        std::sort(parts.begin(), parts.end());
        s.clear();
        for (auto& p : parts) s += p;
        out.emplace_back(s, kv.second);
    }
    std::sort(out.begin(), out.end());
    std::cout << std::setprecision(17);
    std::cout << "{\"mode\": \"uhucpm_cpp_" << o.bound << "\", \"n\": " << out.size() << ", \"patterns\": [";
    for (size_t i = 0; i < out.size(); ++i) std::cout << (i ? ", " : "") << "\"" << json_escape(out[i].first) << "\"";
    std::cout << "]";
    if (scores) {
        std::cout << ", \"scores\": [";
        for (size_t i = 0; i < out.size(); ++i) std::cout << (i ? ", " : "") << out[i].second;
        std::cout << "]";
    }
    std::cout << ", \"seconds\": " << sec << ", \"cliques\": " << pb.C << ", \"evaluated\": " << st.evaluated
              << ", \"visited\": " << st.visited << ", \"Z\": " << pb.Z << ", \"timeout\": " << (st.timeout ? "true" : "false")
              << ", \"max_rss_kb\": " << ru.ru_maxrss << "}" << std::endl;
    return st.timeout ? 3 : 0;
}
