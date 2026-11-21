// Foursquare TSMC2014 check-ins -> instance file (see src/data/foursquare_loader.hpp); JSON summary on stdout.
// Usage: fsq_prep TSV UTM_ZONE NFEAT OUT_CSV  -> OUT_CSV (Feature,Instance,LocX,LocY,a,a_test,venue,category)
#include <iostream>
#include <stdexcept>

#include "common/command_line.hpp"
#include "common/json_writer.hpp"
#include "data/foursquare_loader.hpp"

using namespace colocation;

namespace {

constexpr int kExitUsage = 2;

int run(int argc, char** argv) {
    const CommandLine cl(argc, argv);
    if (cl.positionalCount() < 4) {
        std::cerr << "usage: fsq_prep TSV UTM_ZONE NFEAT OUT_CSV\n";
        return kExitUsage;
    }
    FoursquareSettings settings;
    settings.utmZone = cl.positionalInt(1);
    settings.featureCount = cl.positionalInt(2);
    const FoursquareData data = FoursquareLoader(settings).load(cl.positional(0));
    data.writeInstanceTable(cl.positional(3));
    JsonLineWriter out;
    out.field("n", data.venues.size()).field("features", data.categories.size()).field("Wtr", data.trainingWeeks)
        .field("Wte", data.testWeeks).field("checkins", data.checkins);
    std::cout << out.str() << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "fsq_prep: " << e.what() << "\n";
        return kExitUsage;
    }
}
