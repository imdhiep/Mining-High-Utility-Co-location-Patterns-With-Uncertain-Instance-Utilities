// Scores of given patterns under a posterior: participating instances, total mean M, total variance V (with the
// shared-error term) and Lambda_z = (M - z sqrt(V)) / Z. Used to evaluate reported patterns against the true utilities
// (posterior file with the true values and zero variances) and to check the coverage of the lower bound.
// Usage: pattern_scores DATA_CSV POSTERIOR D Z PATTERNS_FILE
//   PATTERNS_FILE: one pattern per line, feature names concatenated (one-character feature names)
//   --instances   also print the participating instances of every pattern (row numbers of DATA_CSV, from 0)
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

#include "common/command_line.hpp"
#include "common/csv_table.hpp"
#include "common/json_writer.hpp"
#include "common/utility_posterior.hpp"
#include "mining/instance_utilities.hpp"
#include "spatial/clique_table.hpp"
#include "spatial/spatial_dataset.hpp"

using namespace colocation;

namespace {

constexpr int kExitUsage = 2;

std::vector<int> parsePattern(const std::string& text, const std::map<char, int>& index) {
    std::vector<int> features;
    for (char c : text) {
        auto it = index.find(c);
        if (it == index.end()) throw std::invalid_argument(std::string("unknown feature ") + c + " in pattern " + text);
        features.push_back(it->second);
    }
    std::sort(features.begin(), features.end());
    return features;
}

int run(int argc, char** argv) {
    const CommandLine cl(argc, argv, {"instances"});
    if (cl.positionalCount() < 5) {
        std::cerr << "usage: pattern_scores DATA_CSV POSTERIOR D Z PATTERNS_FILE\n";
        return kExitUsage;
    }
    const SpatialDataset data = SpatialDataset::fromTable(CsvTable::read(cl.positional(0)));
    const UtilityPosterior posterior = UtilityPosteriorFile::read(cl.positional(1));
    if (static_cast<int>(posterior.size()) != data.size()) throw std::runtime_error("posterior and data differ in size");
    const CliqueTable table = CliqueTable::build(data, cl.positionalDouble(2));
    const InstanceUtilities utilities(posterior);
    const double z = cl.positionalDouble(3);
    std::map<char, int> index;
    for (int f = 0; f < data.featureCount(); ++f) {
        if (data.featureName(f).size() != 1) throw std::invalid_argument("feature names must have one character");
        index[data.featureName(f)[0]] = f;
    }
    std::ifstream in(cl.positional(4));
    if (!in) throw std::runtime_error("cannot open " + cl.positional(4));
    std::vector<std::string> names;
    std::vector<double> M, V, scores, counts;
    std::vector<JsonLineWriter> members;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const std::vector<int> instances = table.participatingInstances(parsePattern(line, index));
        names.push_back(line);
        M.push_back(utilities.totalMean(instances));
        V.push_back(utilities.totalVariance(instances));
        scores.push_back(utilities.score(instances, z));
        counts.push_back(static_cast<double>(instances.size()));
        if (cl.has("instances")) {
            members.emplace_back();
            members.back().field("instances", std::vector<double>(instances.begin(), instances.end()));
        }
    }
    JsonLineWriter out;
    out.field("patterns", names).field("M", M).field("V", V).field("scores", scores).field("participating", counts)
        .field("Z", utilities.normalizer());
    if (cl.has("instances")) out.field("members", members);
    std::cout << out.str() << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "pattern_scores: " << e.what() << "\n";
        return kExitUsage;
    }
}
