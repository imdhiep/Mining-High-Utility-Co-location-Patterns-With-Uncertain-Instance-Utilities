// Posterior of the count utility (Definition 4) for an instance file with a count column, the bootstrap matrix of the
// total variance and the expected normalizer. Writes a posterior file (io.hpp) for uhucpm_mine.
// Usage: post_counts DATA_CSV COUNT_COLUMN E OUT_POSTERIOR [--t T] [--B B] [--seed S] [--J J] [--iters I]
//                    [--boot-iters I] [--ep FILE]
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>

#include "ceb.hpp"
#include "io.hpp"

using namespace uhucpm;

int main(int argc, char** argv) {
    const auto t0 = std::chrono::steady_clock::now();
    if (argc < 5) { std::cerr << "usage: post_counts DATA_CSV COUNT_COLUMN E OUT [--t T] [--B B] [--seed S] [--J J] [--iters I] [--boot-iters I] [--ep FILE]\n"; return 2; }
    int t = 0, B = 20, J = 300, iters = 300, biters = 40;
    unsigned long long seed = 0;
    std::string ep_path;
    for (int i = 5; i < argc; ++i) {
        std::string a = argv[i];
        if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; return 2; }
        if (a == "--t") t = std::atoi(argv[++i]);
        else if (a == "--B") B = std::atoi(argv[++i]);
        else if (a == "--seed") seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--J") J = std::atoi(argv[++i]);
        else if (a == "--iters") iters = std::atoi(argv[++i]);
        else if (a == "--boot-iters") biters = std::atoi(argv[++i]);
        else if (a == "--ep") ep_path = argv[++i];
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    Table tab = read_csv(argv[1]);
    const int cf = tab.col("Feature"), cc = tab.col(argv[2]);
    const double E = std::atof(argv[3]);
    std::map<std::string, int> fid;
    for (auto& r : tab.rows) fid[r[cf]] = 0;
    int k = 0;
    for (auto& kv : fid) kv.second = k++;
    std::vector<int> F;
    std::vector<double> x;
    for (auto& r : tab.rows) { F.push_back(fid[r[cf]]); x.push_back(std::stod(r[cc])); }
    CebResult R = ceb_posterior(x, E, F, J, iters, t);
    PosteriorFile pf;
    pf.mu = R.mu;
    pf.s2 = R.s2;
    pf.B = B;
    if (B > 0) pf.U = cboot(x, E, R, B, biters, seed);
    pf.Z = 0;
    for (double e : R.ep) pf.Z += e;
    write_posterior(argv[4], pf);
    if (!ep_path.empty()) {
        std::ofstream f(ep_path);
        f << std::setprecision(17);
        for (double e : R.ep) f << e << "\n";
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "{\"n\": " << x.size() << ", \"features\": " << k << ", \"B\": " << B << ", \"Z\": " << std::setprecision(17)
              << pf.Z << ", \"seconds\": " << sec << "}" << std::endl;
    return 0;
}
