// Foursquare TSMC2014 check-ins -> venue instances (port of the loading part of uhucm/fsq_loader.py).
// Input columns: user, venue, category id, category name, latitude, longitude, time-zone offset (minutes), UTC time
// ("Tue Apr 03 18:00:09 +0000 2012"). Local time = UTC time + offset.
//   - generic categories (kExcludedCategories) are dropped;
//   - the features are the `featureCount` categories with the most distinct venues (ties: name order);
//   - each venue takes its most frequent category (ties: the label that appears first in the file);
//   - training period: local time before 2012-12-01, test period: the rest;
//   - instances: venues with at least one training check-in, sorted by venue id, located at the median latitude and
//     longitude of their check-ins and projected to UTM;
//   - exposures: whole days of each period / 7 (weeks).
#pragma once

#include <string>
#include <vector>

namespace colocation {

struct FoursquareSettings {
    int utmZone = 18;
    int featureCount = 20;
};

struct VenueInstance {
    std::string venue;
    std::string category;          // cleaned name (" " and "/" -> "_", "&" -> "and")
    double x = 0.0, y = 0.0;       // UTM
    long long trainingCheckins = 0;
    long long testCheckins = 0;
};

struct FoursquareData {
    std::vector<VenueInstance> venues;   // sorted by venue id
    std::vector<std::string> categories; // sorted cleaned names; feature index = position
    double trainingWeeks = 0.0;
    double testWeeks = 0.0;
    long long checkins = 0;              // check-ins of the selected features

    int featureOf(const VenueInstance& v) const;
    // instance file: Feature,Instance,LocX,LocY,a,a_test,venue,category
    void writeInstanceTable(const std::string& path) const;
};

class FoursquareLoader {
public:
    explicit FoursquareLoader(FoursquareSettings settings) : settings_(settings) {}
    FoursquareData load(const std::string& path) const;

private:
    FoursquareSettings settings_;
};

}  // namespace colocation
