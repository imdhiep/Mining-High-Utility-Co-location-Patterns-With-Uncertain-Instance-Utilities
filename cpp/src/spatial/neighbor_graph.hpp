// Neighbour relation of Section III-A: two instances of different features are neighbours when their distance is at
// most d. Found with a grid of cell size d, each instance checking the 3 x 3 block of cells around its own.
#pragma once

#include <vector>

#include "spatial/spatial_dataset.hpp"

namespace colocation {

class NeighborGraph {
public:
    // squared distances up to d^2 + kDistanceSlack count as neighbours (exact ties at d are kept)
    static constexpr double kDistanceSlack = 1e-9;

    NeighborGraph(const SpatialDataset& data, double distance);

    int size() const { return static_cast<int>(adjacency_.size()); }
    double distance() const { return distance_; }
    // sorted neighbour ids
    const std::vector<int>& neighbors(int i) const { return adjacency_[i]; }
    long long edgeCount() const;

private:
    double distance_;
    std::vector<std::vector<int>> adjacency_;
};

}  // namespace colocation
