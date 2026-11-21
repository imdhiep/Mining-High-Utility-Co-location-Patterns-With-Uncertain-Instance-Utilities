// Yelp businesses, check-ins and reviews -> rated business instances of one or more cities (port of
// uhucm/yelp_full.py, build step).
//   business.json : city, coordinates, categories -> one of 20 features (first match in the feature list)
//   checkin.json  : dated check-ins split at `split` / `testEnd` (or weekday-hour counts in the old release, then
//                   all check-ins count as training)
//   reviews_<City>.csv (review_id,user_id,business_id,stars,date): training reviews (date < split) and test reviews
// A city keeps the businesses whose feature has at least one business with training reviews.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace colocation {

struct YelpSettings {
    std::string split;            // first day of the test period, "YYYY-MM-DD"
    std::string testEnd;          // first day after the test period
};

struct YelpBusiness {
    std::string id;
    std::string feature;          // short feature name (e.g. "Coffee")
    double x = 0.0, y = 0.0;      // UTM
    double trainingCheckins = 0.0;
    double testCheckins = -1.0;   // -1 when the check-ins are not dated
    std::vector<double> trainingStars, testStars;
};

struct YelpCity {
    std::string name;
    int utmZone = 0;
    std::vector<YelpBusiness> businesses;   // in the order of business.json

    // positions of the businesses of the `count` features with the most businesses (ties: name order); all when count <= 0
    std::vector<int> topFeatureBusinesses(int count) const;
};

class YelpLoader {
public:
    explicit YelpLoader(YelpSettings settings) : settings_(std::move(settings)) {}
    // cities in the given order; a city without businesses is left out
    std::vector<YelpCity> load(const std::string& businessJson, const std::string& checkinJson, const std::string& reviewDir,
                               const std::vector<std::string>& cities);
    bool datedCheckins() const { return dated_; }
    static std::string fileName(const std::string& city);   // spaces -> "_"

private:
    struct RawBusiness {
        std::string city, feature;
        double latitude, longitude;
    };

    YelpSettings settings_;
    bool dated_ = false;
    std::vector<std::string> order_;
    std::unordered_map<std::string, RawBusiness> businesses_;
    std::unordered_map<std::string, long long> trainingCheckins_, testCheckins_;
    std::unordered_map<std::string, std::vector<double>> trainingStars_, testStars_;

    void readBusinesses(const std::string& path, const std::vector<std::string>& cities);
    void readCheckins(const std::string& path);
    void readReviews(const std::string& reviewDir, const std::vector<std::string>& cities);
    YelpCity buildCity(const std::string& city) const;
    static std::string featureOf(const std::string& categories);
    static int utmZoneOf(const std::string& city, const std::vector<double>& longitudes);
};

}  // namespace colocation
