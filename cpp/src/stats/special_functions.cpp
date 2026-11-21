#include "stats/special_functions.hpp"

#include <cmath>
#include <stdexcept>

namespace colocation {

namespace {

constexpr double kAsymptoticFrom = 20.0;   // recurrence up to here, then the asymptotic series (error < 1e-15)
constexpr double kPi = 3.14159265358979323846;

}  // namespace

double SpecialFunctions::digamma(double x) {
    if (!(x > 0)) throw std::domain_error("digamma of a non-positive value");
    double r = 0;
    while (x < kAsymptoticFrom) {
        r -= 1 / x;
        x += 1;
    }
    const double f = 1 / (x * x);
    return r + std::log(x) - 0.5 / x - f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f / 132))));
}

double SpecialFunctions::trigamma(double x) {
    if (!(x > 0)) throw std::domain_error("trigamma of a non-positive value");
    double r = 0;
    while (x < kAsymptoticFrom) {
        r += 1 / (x * x);
        x += 1;
    }
    const double f = 1 / (x * x);
    return r + 1 / x + f / 2 + f / x * (1.0 / 6 - f * (1.0 / 30 - f * (1.0 / 42 - f / 30)));
}

double SpecialFunctions::normalCdf(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }

double SpecialFunctions::normalPdf(double z) { return std::exp(-0.5 * z * z) / std::sqrt(2 * kPi); }

}  // namespace colocation
