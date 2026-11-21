// Posterior of the rating utility (equation (4)) for an instance file with star ratings, as yelp_prep computes it for a
// Yelp city, but from the ratings given in the file (semi-synthetic rating data of Section VI-B.2).
// Usage: post_ratings DATA_CSV OUT_POSTERIOR [--B B] [--seed S]
//   DATA_CSV columns: Feature, Instance, LocX, LocY, a (training check-ins, the activity weight ln(1 + a)), stars
//   (the ratings of the training period separated by ';', empty when the business has none)
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "common/command_line.hpp"
#include "common/csv_table.hpp"
#include "common/json_writer.hpp"
#include "common/process_stats.hpp"
#include "common/utility_posterior.hpp"
#include "posterior/rating_posterior_model.hpp"

using namespace colocation;

namespace {

constexpr int kExitUsage = 2;
constexpr int kDefaultBootstrapSamples = 20;

std::vector<double> parseStars(const std::string& text) {
    std::vector<double> out;
    std::stringstream ss(text);
    for (std::string s; std::getline(ss, s, ';');)
        if (!s.empty()) out.push_back(std::stod(s));
    return out;
}

int run(int argc, char** argv) {
    const Stopwatch clock;
    const CommandLine cl(argc, argv);
    if (cl.positionalCount() < 2) {
        std::cerr << "usage: post_ratings DATA_CSV OUT_POSTERIOR [--B B] [--seed S]\n";
        return kExitUsage;
    }
    const int samples = cl.option("B", kDefaultBootstrapSamples);
    const unsigned long long seed = cl.optionUnsigned("seed", 0);
    const CsvTable table = CsvTable::read(cl.positional(0));
    const int cf = table.column("Feature"), ca = table.column("a"), cs = table.column("stars");
    std::map<std::string, int> featureIndex;
    for (const auto& r : table.rows()) featureIndex[r[cf]] = 0;
    int k = 0;
    for (auto& kv : featureIndex) kv.second = k++;
    std::vector<RatedBusiness> businesses;
    for (const auto& r : table.rows())
        businesses.push_back({featureIndex[r[cf]], parseStars(cs < static_cast<int>(r.size()) ? r[cs] : ""), std::stod(r[ca])});
    RatingPosteriorModel model(businesses);
    const PosteriorEstimate estimate = model.estimate();
    UtilityPosterior posterior;
    posterior.mean = estimate.mean;
    posterior.variance = estimate.variance;
    posterior.normalizer = estimate.expectedNormalizer();
    if (samples > 0) posterior.bootstrap = model.bootstrap(samples, seed);
    UtilityPosteriorFile::write(cl.positional(1), posterior);
    JsonLineWriter out;
    out.field("n", businesses.size()).field("features", k).field("B", samples).field("Z", posterior.normalizer)
        .field("sigma_s2", model.pooledVariance()).field("seconds", clock.seconds());
    std::cout << out.str() << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "post_ratings: " << e.what() << "\n";
        return kExitUsage;
    }
}
