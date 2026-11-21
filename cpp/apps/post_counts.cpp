// Posterior of the count utility (Definition 4) for an instance file with a count column: posterior means and
// variances, the bootstrap matrix of the total variance and the expected normalizer, written as a posterior file.
// Usage: post_counts DATA_CSV COUNT_COLUMN E OUT_POSTERIOR [--t T] [--B B] [--seed S] [--J J] [--iters I]
//                    [--boot-iters I] [--ep FILE] [--prior npmle|gamma] [--center moment|median]
//   --ep FILE  also writes E[u+ | data] of every instance, one per line
//   --prior gamma, --center median: the alternatives compared in the experiments (Definition 4 is npmle + moment)
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>

#include "common/command_line.hpp"
#include "common/csv_table.hpp"
#include "common/json_writer.hpp"
#include "common/process_stats.hpp"
#include "common/utility_posterior.hpp"
#include "posterior/count_posterior_model.hpp"
#include "posterior/gamma_poisson_model.hpp"

using namespace colocation;

namespace {

constexpr int kExitUsage = 2;
constexpr int kDefaultBootstrapSamples = 20;

int run(int argc, char** argv) {
    const Stopwatch clock;
    const CommandLine cl(argc, argv);
    if (cl.positionalCount() < 4) {
        std::cerr << "usage: post_counts DATA_CSV COUNT_COLUMN E OUT [--t T] [--B B] [--seed S] [--J J] [--iters I] [--boot-iters I] [--ep FILE]\n";
        return kExitUsage;
    }
    NpmleSettings settings;
    settings.gridPoints = cl.option("J", settings.gridPoints);
    settings.emIterations = cl.option("iters", settings.emIterations);
    settings.bootstrapEmIterations = cl.option("boot-iters", settings.bootstrapEmIterations);
    const int truncation = cl.option("t", 0);
    const int samples = cl.option("B", kDefaultBootstrapSamples);
    const unsigned long long seed = cl.optionUnsigned("seed", 0);

    const CsvTable table = CsvTable::read(cl.positional(0));
    const int cf = table.column("Feature"), cc = table.column(cl.positional(1));
    std::map<std::string, int> featureIndex;
    for (const auto& r : table.rows()) featureIndex[r[cf]] = 0;
    int k = 0;
    for (auto& kv : featureIndex) kv.second = k++;
    std::vector<int> features;
    std::vector<double> counts;
    for (const auto& r : table.rows()) {
        features.push_back(featureIndex[r[cf]]);
        counts.push_back(std::stod(r[cc]));
    }
    const std::string priorName = cl.option("prior", std::string("npmle"));
    if (priorName != "npmle" && priorName != "gamma") throw std::invalid_argument("unknown prior " + priorName);
    if (priorName == "gamma" && truncation > 0) throw std::invalid_argument("the Gamma prior needs t = 0");
    const CenterRule center = parseCenterRule(cl.option("center", std::string("moment")));
    std::unique_ptr<PosteriorModel> model;
    CountPosteriorModel* npmleModel = nullptr;
    if (priorName == "gamma") {
        model = std::make_unique<GammaPoissonPosteriorModel>(counts, features, cl.positionalDouble(2));
    } else {
        auto m = std::make_unique<CountPosteriorModel>(counts, features, cl.positionalDouble(2), truncation, settings, center);
        npmleModel = m.get();
        model = std::move(m);
    }
    const PosteriorEstimate estimate = model->estimate();
    UtilityPosterior posterior;
    posterior.mean = estimate.mean;
    posterior.variance = estimate.variance;
    posterior.normalizer = estimate.expectedNormalizer();
    if (samples > 0) posterior.bootstrap = model->bootstrap(samples, seed);
    UtilityPosteriorFile::write(cl.positional(3), posterior);
    if (cl.has("ep")) {
        std::ofstream ep(cl.option("ep", std::string()));
        ep << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (double e : estimate.expectedPositive) ep << e << "\n";
    }
    JsonLineWriter out;
    out.field("n", counts.size()).field("features", k).field("B", samples).field("Z", posterior.normalizer)
        .field("local_edge_mass_max", npmleModel ? npmleModel->maxLocalEdgeMass() : 0.0).field("prior", priorName)
        .field("seconds", clock.seconds());
    std::cout << out.str() << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "post_counts: " << e.what() << "\n";
        return kExitUsage;
    }
}
