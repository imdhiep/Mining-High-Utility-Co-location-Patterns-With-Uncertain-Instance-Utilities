// Review files of selected Yelp cities (replaces prep_yelp_reviews.py): keeps review_id, user_id, business_id, stars
// and date of the reviews of businesses in the cities, without the review text.
// Usage: yelp_reviews BUSINESS_JSON REVIEW_JSON "City|City|..." OUT_DIR  -> OUT_DIR/reviews_<City>.csv
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "common/command_line.hpp"
#include "common/csv_table.hpp"
#include "common/json_line_parser.hpp"
#include "common/json_writer.hpp"
#include "data/yelp_loader.hpp"

using namespace colocation;

namespace {

constexpr int kExitUsage = 2;

int run(int argc, char** argv) {
    const CommandLine cl(argc, argv);
    if (cl.positionalCount() < 4) {
        std::cerr << "usage: yelp_reviews BUSINESS_JSON REVIEW_JSON \"City|City\" OUT_DIR\n";
        return kExitUsage;
    }
    std::set<std::string> cities;
    std::stringstream ss(cl.positional(2));
    for (std::string c; std::getline(ss, c, '|');) cities.insert(c);
    std::unordered_map<std::string, std::string> cityOf;
    {
        std::ifstream in(cl.positional(0));
        if (!in) throw std::runtime_error("cannot open " + cl.positional(0));
        for (std::string line; std::getline(in, line);) {
            if (line.empty()) continue;
            JsonObject o = JsonLineParser(line).parseObject();
            auto city = o.find("city");
            if (city != o.end() && city->second.isString() && cities.count(city->second.text)) cityOf[o["business_id"].text] = city->second.text;
        }
    }
    std::map<std::string, std::unique_ptr<CsvWriter>> writers;
    std::map<std::string, long long> kept;
    for (const auto& c : cities) {
        writers[c] = std::make_unique<CsvWriter>(cl.positional(3) + "/reviews_" + YelpLoader::fileName(c) + ".csv",
                                                 std::vector<std::string>{"review_id", "user_id", "business_id", "stars", "date"});
        kept[c] = 0;
    }
    std::ifstream in(cl.positional(1));
    if (!in) throw std::runtime_error("cannot open " + cl.positional(1));
    long long read = 0;
    for (std::string line; std::getline(in, line);) {
        if (line.empty()) continue;
        ++read;
        JsonObject o = JsonLineParser(line).parseObject();
        auto it = cityOf.find(o["business_id"].text);
        if (it == cityOf.end()) continue;
        CsvWriter& w = *writers[it->second];
        w.add(o["review_id"].text).add(o["user_id"].text).add(o["business_id"].text).add(o["stars"].number).add(o["date"].text);
        w.endRow();
        ++kept[it->second];
    }
    std::vector<JsonLineWriter> perCity;
    for (const auto& kv : kept) {
        JsonLineWriter c;
        c.field("city", kv.first).field("reviews", kv.second);
        perCity.push_back(std::move(c));
    }
    JsonLineWriter out;
    out.field("businesses", cityOf.size()).field("reviews_read", read).field("cities", perCity);
    std::cout << out.str() << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "yelp_reviews: " << e.what() << "\n";
        return kExitUsage;
    }
}
