#include "geo/utm_projection.hpp"

#include <cmath>

namespace colocation {

namespace {

constexpr double kSemiMajorAxis = 6378137.0;           // WGS84
constexpr double kFlattening = 1.0 / 298.257223563;    // WGS84
constexpr double kScaleFactor = 0.9996;                // UTM
constexpr double kFalseEasting = 500000.0;             // UTM, northern hemisphere: no false northing
constexpr double kPi = 3.14159265358979323846;
constexpr double kZoneWidthDeg = 6.0;

}  // namespace

UtmProjection::UtmProjection(int zone) : zone_(zone), centralMeridianDeg_(kZoneWidthDeg * zone - 183.0) {
    const double n = kFlattening / (2 - kFlattening);
    const double n2 = n * n, n3 = n2 * n, n4 = n3 * n, n5 = n4 * n, n6 = n5 * n;
    rectifyingRadius_ = kSemiMajorAxis / (1 + n) * (1 + n2 / 4 + n4 / 64 + n6 / 256);
    alpha_[0] = 0.0;
    alpha_[1] = n / 2 - 2 * n2 / 3 + 5 * n3 / 16 + 41 * n4 / 180 - 127 * n5 / 288 + 7891 * n6 / 37800;
    alpha_[2] = 13 * n2 / 48 - 3 * n3 / 5 + 557 * n4 / 1440 + 281 * n5 / 630 - 1983433 * n6 / 1935360;
    alpha_[3] = 61 * n3 / 240 - 103 * n4 / 140 + 15061 * n5 / 26880 + 167603 * n6 / 181440;
    alpha_[4] = 49561 * n4 / 161280 - 179 * n5 / 168 + 6601661 * n6 / 7257600;
    alpha_[5] = 34729 * n5 / 80640 - 3418889 * n6 / 1995840;
    alpha_[6] = 212378941 * n6 / 319334400;
    eccentricity_ = std::sqrt(kFlattening * (2 - kFlattening));
}

PlanarPoint UtmProjection::project(double longitudeDeg, double latitudeDeg) const {
    const double phi = latitudeDeg * kPi / 180;
    const double lambda = (longitudeDeg - centralMeridianDeg_) * kPi / 180;
    const double sinPhi = std::sin(phi);
    // conformal latitude through its tangent t, then the transverse Mercator coordinates on the sphere
    const double t = std::sinh(std::atanh(sinPhi) - eccentricity_ * std::atanh(eccentricity_ * sinPhi));
    const double xiPrime = std::atan2(t, std::cos(lambda));
    const double etaPrime = std::atanh(std::sin(lambda) / std::sqrt(1 + t * t));
    double xi = xiPrime, eta = etaPrime;
    for (int j = 1; j <= 6; ++j) {
        xi += alpha_[j] * std::sin(2 * j * xiPrime) * std::cosh(2 * j * etaPrime);
        eta += alpha_[j] * std::cos(2 * j * xiPrime) * std::sinh(2 * j * etaPrime);
    }
    return {kFalseEasting + kScaleFactor * rectifyingRadius_ * eta, kScaleFactor * rectifyingRadius_ * xi};
}

int UtmProjection::zoneOfLongitude(double longitudeDeg) {
    return static_cast<int>(std::floor((longitudeDeg + 180) / kZoneWidthDeg)) + 1;
}

}  // namespace colocation
