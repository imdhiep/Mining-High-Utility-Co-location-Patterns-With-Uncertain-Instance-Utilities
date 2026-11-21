#include "data/foursquare_loader.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>

#include "common/csv_table.hpp"
#include "data/feature_labels.hpp"
#include "geo/utm_projection.hpp"

namespace colocation {

namespace {

const std::set<std::string> kExcludedCategories = {
    "Home (private)", "Residential Building (Apartment / Condo)", "Building", "Road", "Bridge", "Office",
    "Government Building", "College Academic Building", "Other Great Outdoors", "Miscellaneous Shop", "Restaurant",
    "General Entertainment"};

constexpr long long kSecondsPerDay = 86400;
constexpr int kDaysPerWeek = 7;
constexpr int kTrainingEndYear = 2012, kTrainingEndMonth = 12, kTrainingEndDay = 1;
constexpr std::size_t kColumns = 8;

class CivilTime {
public:
    // days since 1970-01-01 (H. Hinnant's algorithm)
    static long long daysFromCivil(long long y, unsigned m, unsigned d) {
        y -= m <= 2;
        const long long era = (y >= 0 ? y : y - 399) / 400;
        const unsigned yoe = static_cast<unsigned>(y - era * 400);
        const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
        const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        return era * 146097 + static_cast<long long>(doe) - 719468;
    }
    // "Tue Apr 03 18:00:09 +0000 2012" -> seconds since the epoch
    static long long parseUtc(const std::string& s) {
        static const char* months[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
        char weekday[4], month[4], zone[6];
        int d, H, M, S, Y;
        if (std::sscanf(s.c_str(), "%3s %3s %d %d:%d:%d %5s %d", weekday, month, &d, &H, &M, &S, zone, &Y) != 8)
            throw std::runtime_error("bad time " + s);
        int m = 0;
        while (m < 12 && std::strcmp(months[m], month)) ++m;
        return daysFromCivil(Y, m + 1, d) * kSecondsPerDay + H * 3600LL + M * 60LL + S;
    }
    static long long floorDiv(long long a, long long b) { return a >= 0 ? a / b : -((-a + b - 1) / b); }
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// the file is read as latin-1, as in fsq_loader
std::string latin1ToUtf8(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (c < 0x80) {
            out += static_cast<char>(c);
        } else {
            out += static_cast<char>(0xC0 | (c >> 6));
            out += static_cast<char>(0x80 | (c & 0x3F));
        }
    }
    return out;
}

std::string cleanCategory(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == ' ' || c == '/') out += '_';
        else if (c == '&') out += "and";
        else out += c;
    }
    return out;
}

struct CheckinRow {
    std::string venue, category;
    double latitude, longitude;
    long long localTime;
};

std::vector<CheckinRow> readCheckins(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<CheckinRow> rows;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::vector<std::string> c = CsvTable::splitLine(line, '\t');
        if (c.size() < kColumns || kExcludedCategories.count(c[3])) continue;
        rows.push_back({c[1], latin1ToUtf8(c[3]), std::stod(c[4]), std::stod(c[5]), CivilTime::parseUtc(c[7]) + 60LL * std::stoll(c[6])});
    }
    return rows;
}

std::set<std::string> topCategories(const std::vector<CheckinRow>& rows, int count) {
    std::map<std::string, std::set<std::string>> venuesOf;
    for (const auto& r : rows) venuesOf[r.category].insert(r.venue);
    std::vector<std::pair<std::string, std::size_t>> sizes;
    for (const auto& kv : venuesOf) sizes.push_back({kv.first, kv.second.size()});
    std::stable_sort(sizes.begin(), sizes.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::set<std::string> top;
    for (int i = 0; i < count && i < static_cast<int>(sizes.size()); ++i) top.insert(sizes[i].first);
    return top;
}

// most frequent label per venue; ties go to the label that appears first in the file (pandas value_counts order)
std::map<std::string, std::string> venueCategories(const std::vector<CheckinRow>& rows) {
    std::map<std::string, std::vector<std::pair<std::string, int>>> labels;
    for (const auto& r : rows) {
        auto& list = labels[r.venue];
        auto it = std::find_if(list.begin(), list.end(), [&](const auto& p) { return p.first == r.category; });
        if (it == list.end()) list.push_back({r.category, 1});
        else ++it->second;
    }
    std::map<std::string, std::string> out;
    for (const auto& kv : labels) {
        int best = -1;
        for (const auto& c : kv.second)
            if (c.second > best) {
                best = c.second;
                out[kv.first] = c.first;
            }
    }
    return out;
}

}  // namespace

int FoursquareData::featureOf(const VenueInstance& v) const {
    return static_cast<int>(std::lower_bound(categories.begin(), categories.end(), v.category) - categories.begin());
}

void FoursquareData::writeInstanceTable(const std::string& path) const {
    CsvWriter out(path, {"Feature", "Instance", "LocX", "LocY", "a", "a_test", "venue", "category"});
    std::map<int, long long> counter;
    for (const auto& v : venues) {
        const int f = featureOf(v);
        out.add(std::string(1, FeatureLabels::label(f))).add(++counter[f]).add(v.x).add(v.y).add(v.trainingCheckins)
            .add(v.testCheckins).add(v.venue).add(v.category);
        out.endRow();
    }
}

FoursquareData FoursquareLoader::load(const std::string& path) const {
    const std::vector<CheckinRow> all = readCheckins(path);
    const std::set<std::string> top = topCategories(all, settings_.featureCount);
    std::vector<CheckinRow> rows;
    for (const auto& r : all)
        if (top.count(r.category)) rows.push_back(r);
    if (rows.empty()) throw std::runtime_error("no check-ins in the selected categories");
    const std::map<std::string, std::string> category = venueCategories(rows);

    const long long trainingEnd = CivilTime::daysFromCivil(kTrainingEndYear, kTrainingEndMonth, kTrainingEndDay) * kSecondsPerDay;
    long long first = rows[0].localTime, last = rows[0].localTime;
    std::map<std::string, long long> training, test;
    std::map<std::string, std::vector<double>> latitudes, longitudes;
    for (const auto& r : rows) {
        first = std::min(first, r.localTime);
        last = std::max(last, r.localTime);
        (r.localTime < trainingEnd ? training : test)[r.venue]++;
        latitudes[r.venue].push_back(r.latitude);
        longitudes[r.venue].push_back(r.longitude);
    }
    FoursquareData data;
    data.trainingWeeks = CivilTime::floorDiv(trainingEnd - first, kSecondsPerDay) / static_cast<double>(kDaysPerWeek);
    data.testWeeks = CivilTime::floorDiv(last - trainingEnd, kSecondsPerDay) / static_cast<double>(kDaysPerWeek);
    data.checkins = static_cast<long long>(rows.size());

    const UtmProjection utm(settings_.utmZone);
    std::set<std::string> names;
    for (const auto& kv : training) {   // map order: venues sorted by id
        const std::string& venue = kv.first;
        const PlanarPoint p = utm.project(median(longitudes[venue]), median(latitudes[venue]));
        const auto t = test.find(venue);
        VenueInstance v{venue, cleanCategory(category.at(venue)), p.x, p.y, kv.second, t == test.end() ? 0 : t->second};
        names.insert(v.category);
        data.venues.push_back(std::move(v));
    }
    data.categories.assign(names.begin(), names.end());
    if (static_cast<int>(data.categories.size()) > FeatureLabels::capacity()) throw std::runtime_error("too many features");
    return data;
}

}  // namespace colocation
