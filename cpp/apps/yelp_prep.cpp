// Yelp cities -> instance files and rating posteriors (equation (4)); see src/data/yelp_loader.hpp and
// src/posterior/rating_posterior_model.hpp. As in uhucm/yelp_full.py, the posterior means and variances are estimated on
// all businesses of the city, then the optional --nfeat selection is applied, and the bootstrap of the total variance
// and the expected normalizer use the selected businesses only.
// Usage: yelp_prep BUSINESS_JSON CHECKIN_JSON REVIEW_DIR "City|City|..." SPLIT TEST_END OUT_DIR [--nfeat K] [--B B] [--seed S]
//   -> OUT_DIR/<City>.csv and OUT_DIR/<City>.post, JSON summary on stdout
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include "common/command_line.hpp"
#include "common/csv_table.hpp"
#include "common/json_writer.hpp"
#include "common/utility_posterior.hpp"
#include "data/feature_labels.hpp"
#include "data/yelp_loader.hpp"
#include "posterior/rating_posterior_model.hpp"

using namespace colocation;

namespace {

constexpr int kExitUsage = 2;
constexpr int kDefaultBootstrapSamples = 20;

double sumOf(const std::vector<double>& v, bool squares) {
    double s = 0;
    for (double x : v) s += squares ? x * x : x;
    return s;
}

// feature indices by sorted feature name over the given businesses
std::vector<int> featureIndices(const YelpCity& city, const std::vector<int>& rows, std::vector<std::string>& names) {
    std::set<std::string> set;
    for (int i : rows) set.insert(city.businesses[i].feature);
    names.assign(set.begin(), set.end());
    std::map<std::string, int> index;
    for (std::size_t j = 0; j < names.size(); ++j) index[names[j]] = static_cast<int>(j);
    std::vector<int> out;
    for (int i : rows) out.push_back(index[city.businesses[i].feature]);
    return out;
}

std::vector<RatedBusiness> ratedBusinesses(const YelpCity& city, const std::vector<int>& rows, const std::vector<int>& features) {
    std::vector<RatedBusiness> out;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        const YelpBusiness& b = city.businesses[rows[r]];
        out.push_back({features[r], b.trainingStars, b.trainingCheckins});
    }
    return out;
}

JsonLineWriter processCity(const YelpCity& city, int featureCount, int samples, unsigned long long seed, bool dated,
                           const std::string& outDir) {
    std::vector<int> all(city.businesses.size());
    for (std::size_t i = 0; i < all.size(); ++i) all[i] = static_cast<int>(i);
    std::vector<std::string> allNames;
    RatingPosteriorModel cityModel(ratedBusinesses(city, all, featureIndices(city, all, allNames)));
    const PosteriorEstimate estimate = cityModel.estimate();

    const std::vector<int> kept = city.topFeatureBusinesses(featureCount);
    std::vector<std::string> names;
    const std::vector<int> features = featureIndices(city, kept, names);
    RatingPosteriorModel keptModel(ratedBusinesses(city, kept, features));
    UtilityPosterior posterior;
    for (int i : kept) {
        posterior.mean.push_back(estimate.mean[i]);
        posterior.variance.push_back(estimate.variance[i]);
        posterior.normalizer += estimate.expectedPositive[i];
    }
    posterior.bootstrap = keptModel.bootstrap(samples, seed);

    const RatingSettings settings;
    const std::string base = outDir + "/" + YelpLoader::fileName(city.name);
    CsvWriter csv(base + ".csv", {"Feature", "Instance", "LocX", "LocY", "a", "a_test", "n_r", "S", "SS", "n_test", "mu", "s2",
                                  "mu_test", "business_id", "feat"});
    std::map<int, long long> counter;
    for (std::size_t r = 0; r < kept.size(); ++r) {
        const YelpBusiness& b = city.businesses[kept[r]];
        const double testCheckins = dated ? b.testCheckins : b.trainingCheckins;
        csv.add(std::string(1, FeatureLabels::label(features[r]))).add(++counter[features[r]]).add(b.x).add(b.y)
            .add(b.trainingCheckins).add(b.testCheckins).add(static_cast<double>(b.trainingStars.size()))
            .add(sumOf(b.trainingStars, false)).add(sumOf(b.trainingStars, true)).add(static_cast<double>(b.testStars.size()))
            .add(posterior.mean[r]).add(posterior.variance[r])
            .add(RatingPosteriorModel::testUtility(testCheckins, b.testStars, settings)).add(b.id).add(b.feature);
        csv.endRow();
    }
    UtilityPosteriorFile::write(base + ".post", posterior);
    JsonLineWriter summary;
    summary.field("city", city.name).field("n", kept.size()).field("features", names.size())
        .field("sigma_s2", cityModel.pooledVariance()).field("zone", city.utmZone).field("Z", posterior.normalizer);
    return summary;
}

int run(int argc, char** argv) {
    const CommandLine cl(argc, argv);
    if (cl.positionalCount() < 7) {
        std::cerr << "usage: yelp_prep BUSINESS CHECKIN REVIEW_DIR CITIES SPLIT TEST_END OUT_DIR [--nfeat K] [--B B] [--seed S]\n";
        return kExitUsage;
    }
    std::vector<std::string> cityNames;
    std::stringstream ss(cl.positional(3));
    for (std::string c; std::getline(ss, c, '|');) cityNames.push_back(c);
    YelpLoader loader({cl.positional(4), cl.positional(5)});
    const std::vector<YelpCity> cities = loader.load(cl.positional(0), cl.positional(1), cl.positional(2), cityNames);
    std::vector<JsonLineWriter> summaries;
    for (const auto& city : cities)
        summaries.push_back(processCity(city, cl.option("nfeat", 0), cl.option("B", kDefaultBootstrapSamples),
                                        cl.optionUnsigned("seed", 0), loader.datedCheckins(), cl.positional(6)));
    JsonLineWriter out;
    out.field("dated_checkins", loader.datedCheckins()).field("cities", summaries);
    std::cout << out.str() << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "yelp_prep: " << e.what() << "\n";
        return kExitUsage;
    }
}
