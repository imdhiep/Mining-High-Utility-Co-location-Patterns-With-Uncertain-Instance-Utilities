// Small input/output helpers shared by the C++ programs.
#pragma once
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace uhucpm {

// split one CSV line on commas (no quoting: the instance files of all methods are plain numbers and short names)
inline std::vector<std::string> split_csv(const std::string& line, char sep = ',') {
    std::vector<std::string> out;
    std::string cell;
    std::stringstream ss(line);
    while (std::getline(ss, cell, sep)) out.push_back(cell);
    if (!line.empty() && line.back() == sep) out.push_back("");
    return out;
}

struct Table {
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;
    int col(const std::string& name) const {
        for (size_t i = 0; i < header.size(); ++i)
            if (header[i] == name) return (int)i;
        throw std::runtime_error("missing column " + name);
    }
};

inline Table read_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    Table t;
    std::string line;
    bool first = true;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (first) { t.header = split_csv(line); first = false; continue; }
        t.rows.push_back(split_csv(line));
    }
    return t;
}

// posterior file: int64 n, int64 B, double Z, then n means, n variances and B x n bootstrap posterior means (row major)
struct PosteriorFile {
    std::vector<double> mu, s2, U;
    int64_t n = 0, B = 0;
    double Z = 0;
};

inline PosteriorFile read_posterior(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    PosteriorFile p;
    f.read(reinterpret_cast<char*>(&p.n), 8);
    f.read(reinterpret_cast<char*>(&p.B), 8);
    f.read(reinterpret_cast<char*>(&p.Z), 8);
    p.mu.resize(p.n);
    p.s2.resize(p.n);
    p.U.resize((size_t)p.B * p.n);
    f.read(reinterpret_cast<char*>(p.mu.data()), 8 * p.n);
    f.read(reinterpret_cast<char*>(p.s2.data()), 8 * p.n);
    if (p.B > 0) f.read(reinterpret_cast<char*>(p.U.data()), 8 * p.B * p.n);
    if (!f) throw std::runtime_error("truncated posterior file " + path);
    return p;
}

inline void write_posterior(const std::string& path, const PosteriorFile& p) {
    std::ofstream f(path, std::ios::binary);
    int64_t n = (int64_t)p.mu.size(), B = p.B;
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(reinterpret_cast<const char*>(&B), 8);
    f.write(reinterpret_cast<const char*>(&p.Z), 8);
    f.write(reinterpret_cast<const char*>(p.mu.data()), 8 * n);
    f.write(reinterpret_cast<const char*>(p.s2.data()), 8 * n);
    if (B > 0) f.write(reinterpret_cast<const char*>(p.U.data()), 8 * B * n);
}

inline std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}

}  // namespace uhucpm
