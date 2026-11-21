#include "data/yelp_loader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include "common/csv_table.hpp"
#include "common/json_line_parser.hpp"
#include "geo/utm_projection.hpp"

namespace colocation {

namespace {

// Yelp category -> feature name; a business takes the first listed category it has
const std::vector<std::pair<std::string, std::string>> kFeatures = {
    {"Coffee & Tea", "Coffee"}, {"Fast Food", "FastFood"}, {"Pizza", "Pizza"}, {"Mexican", "Mexican"}, {"Chinese", "Chinese"},
    {"Japanese", "Japanese"}, {"Italian", "Italian"}, {"Burgers", "Burgers"}, {"Breakfast & Brunch", "Breakfast"},
    {"Desserts", "Desserts"}, {"Bars", "Bars"}, {"Hotels", "Hotels"}, {"Nail Salons", "Nails"}, {"Hair Salons", "Hair"},
    {"Fitness & Instruction", "Fitness"}, {"Auto Repair", "AutoRepair"}, {"Dentists", "Dentists"}, {"Grocery", "Grocery"},
    {"Banks & Credit Unions", "Banks"}, {"Fashion", "Fashion"}};

// zones fixed by the earlier experiments for cities that straddle a zone border
const std::map<std::string, int> kCityZones = {{"Las Vegas", 11}, {"Phoenix", 12}, {"Toronto", 17}, {"Charlotte", 17},
                                               {"Pittsburgh", 17}, {"Calgary", 11}, {"Scottsdale", 12}, {"Mesa", 12}};

constexpr std::size_t kDateLength = 10;   // "YYYY-MM-DD"

std::string strip(const std::string& s) {   // Python str.strip()
    std::size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

}  // namespace

std::vector<int> YelpCity::topFeatureBusinesses(int count) const {
    std::vector<int> all(businesses.size());
    for (std::size_t i = 0; i < businesses.size(); ++i) all[i] = static_cast<int>(i);
    if (count <= 0) return all;
    std::map<std::string, int> sizes;
    for (const auto& b : businesses) sizes[b.feature]++;
    std::vector<std::pair<std::string, int>> order(sizes.begin(), sizes.end());
    std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::set<std::string> keep;
    for (int i = 0; i < count && i < static_cast<int>(order.size()); ++i) keep.insert(order[i].first);
    std::vector<int> out;
    for (int i : all)
        if (keep.count(businesses[i].feature)) out.push_back(i);
    return out;
}

std::string YelpLoader::fileName(const std::string& city) {
    std::string out = city;
    std::replace(out.begin(), out.end(), ' ', '_');
    return out;
}

std::string YelpLoader::featureOf(const std::string& categories) {
    std::set<std::string> listed;
    std::stringstream ss(categories);
    std::string c;
    while (std::getline(ss, c, ',')) listed.insert(strip(c));
    if (!categories.empty() && categories.back() == ',') listed.insert("");
    for (const auto& kv : kFeatures)
        if (listed.count(kv.first)) return kv.second;
    return "";
}

int YelpLoader::utmZoneOf(const std::string& city, const std::vector<double>& longitudes) {
    auto it = kCityZones.find(city);
    if (it != kCityZones.end()) return it->second;
    std::vector<double> sorted = longitudes;
    std::sort(sorted.begin(), sorted.end());
    return UtmProjection::zoneOfLongitude(sorted[sorted.size() / 2]);
}

void YelpLoader::readBusinesses(const std::string& path, const std::vector<std::string>& cities) {
    const std::set<std::string> wanted(cities.begin(), cities.end());
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        JsonObject o = JsonLineParser(line).parseObject();
        auto city = o.find("city");
        if (city == o.end() || !city->second.isString() || !wanted.count(city->second.text)) continue;
        const std::string id = o["business_id"].text;
        if (businesses_.count(id)) continue;
        auto lat = o.find("latitude");
        if (lat == o.end() || !lat->second.isNumber()) continue;
        auto cat = o.find("categories");
        const std::string feature = featureOf(cat != o.end() && cat->second.isString() ? cat->second.text : "");
        if (feature.empty()) continue;
        businesses_[id] = {city->second.text, feature, lat->second.number, o["longitude"].number};
        order_.push_back(id);
    }
}

void YelpLoader::readCheckins(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        JsonObject o = JsonLineParser(line).parseObject();
        const std::string id = o["business_id"].text;
        if (!businesses_.count(id)) continue;
        auto date = o.find("date");
        if (date != o.end()) {
            dated_ = true;
            std::stringstream ss(date->second.text);
            std::string t;
            while (std::getline(ss, t, ',')) {
                const std::string day = strip(t).substr(0, kDateLength);
                if (day < settings_.split) trainingCheckins_[id]++;
                else if (day < settings_.testEnd) testCheckins_[id]++;
            }
            continue;
        }
        auto time = o.find("time");   // old release: {"weekday-hour": count, ...}
        if (time != o.end() && time->second.kind == JsonValue::Kind::Raw) {
            const std::string raw = time->second.text;
            long long total = 0;
            for (const auto& kv : JsonLineParser(raw).parseObject()) total += static_cast<long long>(kv.second.number);
            trainingCheckins_[id] += total;
        }
    }
}

void YelpLoader::readReviews(const std::string& reviewDir, const std::vector<std::string>& cities) {
    for (const auto& city : cities) {
        const CsvTable t = CsvTable::read(reviewDir + "/reviews_" + fileName(city) + ".csv");
        const int cb = t.column("business_id"), cs = t.column("stars"), cd = t.column("date");
        for (const auto& r : t.rows()) {
            if (!businesses_.count(r[cb])) continue;
            const std::string day = r[cd].substr(0, kDateLength);
            if (day < settings_.split) trainingStars_[r[cb]].push_back(std::stod(r[cs]));
            else if (day < settings_.testEnd) testStars_[r[cb]].push_back(std::stod(r[cs]));
        }
    }
}

YelpCity YelpLoader::buildCity(const std::string& city) const {
    YelpCity out;
    out.name = city;
    std::vector<std::string> ids;
    for (const auto& id : order_)
        if (businesses_.at(id).city == city) ids.push_back(id);
    if (ids.empty()) return out;
    std::vector<double> longitudes;
    for (const auto& id : ids) longitudes.push_back(businesses_.at(id).longitude);
    out.utmZone = utmZoneOf(city, longitudes);
    // features with at least one business with training reviews
    std::set<std::string> rated;
    auto starsOf = [](const std::unordered_map<std::string, std::vector<double>>& m, const std::string& id) {
        auto it = m.find(id);
        return it == m.end() ? std::vector<double>{} : it->second;
    };
    auto countOf = [](const std::unordered_map<std::string, long long>& m, const std::string& id) {
        auto it = m.find(id);
        return it == m.end() ? 0LL : it->second;
    };
    for (const auto& id : ids)
        if (!starsOf(trainingStars_, id).empty()) rated.insert(businesses_.at(id).feature);
    const UtmProjection utm(out.utmZone);
    for (const auto& id : ids) {
        const RawBusiness& b = businesses_.at(id);
        if (!rated.count(b.feature)) continue;
        const PlanarPoint p = utm.project(b.longitude, b.latitude);
        YelpBusiness y;
        y.id = id;
        y.feature = b.feature;
        y.x = p.x;
        y.y = p.y;
        y.trainingCheckins = static_cast<double>(countOf(trainingCheckins_, id));
        y.testCheckins = dated_ ? static_cast<double>(countOf(testCheckins_, id)) : -1.0;
        y.trainingStars = starsOf(trainingStars_, id);
        y.testStars = starsOf(testStars_, id);
        out.businesses.push_back(std::move(y));
    }
    return out;
}

std::vector<YelpCity> YelpLoader::load(const std::string& businessJson, const std::string& checkinJson,
                                       const std::string& reviewDir, const std::vector<std::string>& cities) {
    readBusinesses(businessJson, cities);
    readCheckins(checkinJson);
    readReviews(reviewDir, cities);
    std::vector<YelpCity> out;
    for (const auto& city : cities) {
        YelpCity c = buildCity(city);
        if (!c.businesses.empty()) out.push_back(std::move(c));
    }
    return out;
}

}  // namespace colocation
