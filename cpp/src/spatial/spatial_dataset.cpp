#include "spatial/spatial_dataset.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace colocation {

SpatialDataset::SpatialDataset(std::vector<SpatialInstance> instances, std::vector<std::string> featureNames)
    : instances_(std::move(instances)), featureNames_(std::move(featureNames)) {
    for (const auto& o : instances_)
        if (o.feature < 0 || o.feature >= static_cast<int>(featureNames_.size()))
            throw std::invalid_argument("instance with an unknown feature");
}

SpatialDataset SpatialDataset::fromTable(const CsvTable& table) {
    const int cf = table.column("Feature"), cx = table.column("LocX"), cy = table.column("LocY");
    std::map<std::string, int> index;
    for (const auto& row : table.rows()) index[row[cf]] = 0;
    std::vector<std::string> names;
    for (auto& kv : index) {
        kv.second = static_cast<int>(names.size());
        names.push_back(kv.first);
    }
    std::vector<SpatialInstance> instances;
    instances.reserve(table.size());
    for (const auto& row : table.rows()) instances.push_back({index[row[cf]], std::stod(row[cx]), std::stod(row[cy])});
    return SpatialDataset(std::move(instances), std::move(names));
}

std::string SpatialDataset::patternName(const std::vector<int>& features) const {
    std::vector<std::string> parts;
    for (int f : features) parts.push_back(featureNames_[f]);
    std::sort(parts.begin(), parts.end());
    std::string name;
    for (const auto& p : parts) name += p;
    return name;
}

}  // namespace colocation
