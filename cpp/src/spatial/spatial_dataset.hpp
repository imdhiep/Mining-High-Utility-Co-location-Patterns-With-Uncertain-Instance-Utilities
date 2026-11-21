// Spatial instances with their features (Section III-A): every instance has a feature index and planar coordinates.
#pragma once

#include <string>
#include <vector>

#include "common/csv_table.hpp"

namespace colocation {

struct SpatialInstance {
    int feature = 0;
    double x = 0.0;
    double y = 0.0;
};

class SpatialDataset {
public:
    SpatialDataset() = default;
    SpatialDataset(std::vector<SpatialInstance> instances, std::vector<std::string> featureNames);

    // instance file with columns Feature, LocX, LocY; feature indices follow the sorted feature names
    static SpatialDataset fromTable(const CsvTable& table);

    int size() const { return static_cast<int>(instances_.size()); }
    int featureCount() const { return static_cast<int>(featureNames_.size()); }
    const SpatialInstance& instance(int i) const { return instances_[i]; }
    const std::vector<SpatialInstance>& instances() const { return instances_; }
    const std::string& featureName(int f) const { return featureNames_[f]; }
    // pattern name: the names of its features sorted and concatenated
    std::string patternName(const std::vector<int>& features) const;

private:
    std::vector<SpatialInstance> instances_;
    std::vector<std::string> featureNames_;
};

}  // namespace colocation
