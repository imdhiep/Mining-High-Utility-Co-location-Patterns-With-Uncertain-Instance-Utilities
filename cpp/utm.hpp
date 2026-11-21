// WGS84 geographic coordinates -> UTM (northern hemisphere), Krueger series to sixth order in n (Karney 2011),
// accurate to a few nanometres inside a zone; used in place of pyproj (EPSG:326ZZ).
#pragma once
#include <cmath>

namespace uhucpm {

inline void utm_north(double lon_deg, double lat_deg, int zone, double& E, double& N) {
    const double a = 6378137.0, f = 1.0 / 298.257223563, k0 = 0.9996;
    const double n = f / (2 - f), n2 = n * n, n3 = n2 * n, n4 = n3 * n, n5 = n4 * n, n6 = n5 * n;
    const double A = a / (1 + n) * (1 + n2 / 4 + n4 / 64 + n6 / 256);
    const double al[7] = {0,
                          n / 2 - 2 * n2 / 3 + 5 * n3 / 16 + 41 * n4 / 180 - 127 * n5 / 288 + 7891 * n6 / 37800,
                          13 * n2 / 48 - 3 * n3 / 5 + 557 * n4 / 1440 + 281 * n5 / 630 - 1983433 * n6 / 1935360,
                          61 * n3 / 240 - 103 * n4 / 140 + 15061 * n5 / 26880 + 167603 * n6 / 181440,
                          49561 * n4 / 161280 - 179 * n5 / 168 + 6601661 * n6 / 7257600,
                          34729 * n5 / 80640 - 3418889 * n6 / 1995840,
                          212378941 * n6 / 319334400};
    const double e = std::sqrt(f * (2 - f));
    const double pi = 3.14159265358979323846;
    const double phi = lat_deg * pi / 180, lam = (lon_deg - (6.0 * zone - 183.0)) * pi / 180;
    const double sp = std::sin(phi);
    const double t = std::sinh(std::atanh(sp) - e * std::atanh(e * sp));
    const double xi1 = std::atan2(t, std::cos(lam)), eta1 = std::atanh(std::sin(lam) / std::sqrt(1 + t * t));
    double xi = xi1, eta = eta1;
    for (int j = 1; j <= 6; ++j) {
        xi += al[j] * std::sin(2 * j * xi1) * std::cosh(2 * j * eta1);
        eta += al[j] * std::cos(2 * j * xi1) * std::sinh(2 * j * eta1);
    }
    E = 500000.0 + k0 * A * eta;
    N = k0 * A * xi;
}

}  // namespace uhucpm
