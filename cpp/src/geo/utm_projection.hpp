// WGS84 longitude/latitude -> UTM easting/northing in a northern zone (EPSG:326ZZ), Krueger series to sixth order in
// the third flattening (Karney, J. Geodesy 85, 2011), accurate to a few nanometres inside a zone.
#pragma once

namespace colocation {

struct PlanarPoint {
    double x = 0.0;
    double y = 0.0;
};

class UtmProjection {
public:
    explicit UtmProjection(int zone);
    PlanarPoint project(double longitudeDeg, double latitudeDeg) const;
    int zone() const { return zone_; }
    // zone whose central meridian is nearest to a longitude
    static int zoneOfLongitude(double longitudeDeg);

private:
    int zone_;
    double centralMeridianDeg_;
    double rectifyingRadius_;          // A
    double alpha_[7];                  // Krueger coefficients alpha_1..alpha_6
    double eccentricity_;
};

}  // namespace colocation
