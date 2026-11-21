// Maximal cliques of the neighbour graph and the clique table of Section V-B: one row per maximal clique, one column
// per feature, entry (Q, f) = the instance of feature f in clique Q or kNone. Instances of one feature are never
// neighbours, so a clique holds at most one instance per feature.
#pragma once

#include <cstdint>
#include <vector>

#include "spatial/neighbor_graph.hpp"

namespace colocation {

// Bron-Kerbosch with pivoting, vertices taken in degeneracy order (Eppstein, Loeffler and Strash 2010);
// returns the maximal cliques with at least two vertices, each sorted, the list sorted
class MaximalCliqueEnumerator {
public:
    static std::vector<std::vector<int>> enumerate(const NeighborGraph& graph);

private:
    static std::vector<int> degeneracyOrder(const NeighborGraph& graph);
};

class CliqueTable {
public:
    static constexpr int32_t kNone = -1;

    CliqueTable(const SpatialDataset& data, const std::vector<std::vector<int>>& cliques);
    // convenience: neighbour graph and maximal cliques of a dataset
    static CliqueTable build(const SpatialDataset& data, double distance);

    int rows() const { return rows_; }
    int features() const { return features_; }
    int instanceCount() const { return instances_; }
    int32_t at(int row, int feature) const { return members_[static_cast<std::size_t>(row) * features_ + feature]; }
    bool has(int row, int feature) const { return at(row, feature) != kNone; }
    // rows that contain an instance of the feature
    const std::vector<int>& rowsWithFeature(int feature) const { return rowsWithFeature_[feature]; }
    // participating instances of a pattern (sorted feature indices): instances of its features in the rows holding all
    std::vector<int> participatingInstances(const std::vector<int>& pattern) const;

private:
    int rows_ = 0;
    int features_ = 0;
    int instances_ = 0;
    std::vector<int32_t> members_;
    std::vector<std::vector<int>> rowsWithFeature_;
};

}  // namespace colocation
