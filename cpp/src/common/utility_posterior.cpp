#include "common/utility_posterior.hpp"

#include <fstream>
#include <stdexcept>

namespace colocation {

namespace {

template <typename T>
void readValues(std::ifstream& in, T* data, std::size_t count) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(sizeof(T) * count));
}

template <typename T>
void writeValues(std::ofstream& out, const T* data, std::size_t count) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(sizeof(T) * count));
}

}  // namespace

UtilityPosterior UtilityPosteriorFile::read(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    int64_t n = 0, samples = 0;
    UtilityPosterior p;
    readValues(in, &n, 1);
    readValues(in, &samples, 1);
    readValues(in, &p.normalizer, 1);
    p.mean.resize(static_cast<std::size_t>(n));
    p.variance.resize(static_cast<std::size_t>(n));
    readValues(in, p.mean.data(), p.mean.size());
    readValues(in, p.variance.data(), p.variance.size());
    p.bootstrap = BootstrapMatrix(static_cast<int>(samples), static_cast<int>(n));
    if (samples > 0) readValues(in, p.bootstrap.values().data(), p.bootstrap.values().size());
    if (!in) throw std::runtime_error("truncated posterior file " + path);
    return p;
}

void UtilityPosteriorFile::write(const std::string& path, const UtilityPosterior& p) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    const int64_t n = static_cast<int64_t>(p.mean.size()), samples = p.bootstrap.samples();
    writeValues(out, &n, 1);
    writeValues(out, &samples, 1);
    writeValues(out, &p.normalizer, 1);
    writeValues(out, p.mean.data(), p.mean.size());
    writeValues(out, p.variance.data(), p.variance.size());
    if (samples > 0) writeValues(out, p.bootstrap.values().data(), p.bootstrap.values().size());
}

}  // namespace colocation
