#include "spatial/neighbor_graph.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace colocation {

namespace {

long long cellKey(long long gx, long long gy) { return (gx << 32) ^ (gy & 0xffffffffLL); }

}  // namespace

NeighborGraph::NeighborGraph(const SpatialDataset& data, double distance) : distance_(distance) {
    const int n = data.size();
    adjacency_.assign(n, {});
    std::unordered_map<long long, std::vector<int>> grid;
    std::vector<long long> cx(n), cy(n);
    for (int i = 0; i < n; ++i) {
        cx[i] = static_cast<long long>(std::floor(data.instance(i).x / distance));
        cy[i] = static_cast<long long>(std::floor(data.instance(i).y / distance));
        grid[cellKey(cx[i], cy[i])].push_back(i);
    }
    const double limit = distance * distance + kDistanceSlack;
    for (int i = 0; i < n; ++i) {
        const SpatialInstance& a = data.instance(i);
        for (int dx = -1; dx <= 1; ++dx)
            for (int dy = -1; dy <= 1; ++dy) {
                auto cell = grid.find(cellKey(cx[i] + dx, cy[i] + dy));
                if (cell == grid.end()) continue;
                for (int j : cell->second) {
                    const SpatialInstance& b = data.instance(j);
                    if (j == i || b.feature == a.feature) continue;
                    const double ex = b.x - a.x, ey = b.y - a.y;
                    if (ex * ex + ey * ey <= limit) adjacency_[i].push_back(j);
                }
            }
        std::sort(adjacency_[i].begin(), adjacency_[i].end());
        adjacency_[i].erase(std::unique(adjacency_[i].begin(), adjacency_[i].end()), adjacency_[i].end());
    }
}

long long NeighborGraph::edgeCount() const {
    long long twice = 0;
    for (const auto& a : adjacency_) twice += static_cast<long long>(a.size());
    return twice / 2;
}

}  // namespace colocation
