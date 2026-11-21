#include "spatial/clique_table.hpp"

#include <algorithm>
#include <iterator>

namespace colocation {

namespace {

std::vector<int> intersect(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> out;
    out.reserve(std::min(a.size(), b.size()));
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
    return out;
}

// bucket queue of vertices keyed by their current degree, with O(1) removal
class DegreeBuckets {
public:
    DegreeBuckets(int n, int maxDegree) : buckets_(maxDegree + 1), position_(n, -1), degree_(n, 0) {}
    void insert(int v, int degree) {
        degree_[v] = degree;
        position_[v] = static_cast<int>(buckets_[degree].size());
        buckets_[degree].push_back(v);
    }
    void remove(int v) {
        auto& bucket = buckets_[degree_[v]];
        const int last = bucket.back();
        bucket[position_[v]] = last;
        position_[last] = position_[v];
        bucket.pop_back();
    }
    void decrement(int v) {
        remove(v);
        insert(v, degree_[v] - 1);
    }
    // removes and returns a vertex of smallest degree
    int popMinimum() {
        int d = 0;
        while (buckets_[d].empty()) ++d;
        const int v = buckets_[d].back();
        remove(v);
        return v;
    }

private:
    std::vector<std::vector<int>> buckets_;
    std::vector<int> position_, degree_;
};

struct SearchFrame {
    std::vector<int> clique, candidates, excluded;   // R, P, X of Bron-Kerbosch
};

}  // namespace

std::vector<int> MaximalCliqueEnumerator::degeneracyOrder(const NeighborGraph& g) {
    const int n = g.size();
    std::vector<int> vertices;
    int maxDegree = 0;
    for (int v = 0; v < n; ++v)
        if (!g.neighbors(v).empty()) {
            vertices.push_back(v);
            maxDegree = std::max(maxDegree, static_cast<int>(g.neighbors(v).size()));
        }
    DegreeBuckets buckets(n, maxDegree);
    for (int v : vertices) buckets.insert(v, static_cast<int>(g.neighbors(v).size()));
    std::vector<char> removed(n, 0);
    std::vector<int> order;
    order.reserve(vertices.size());
    for (std::size_t step = 0; step < vertices.size(); ++step) {
        const int v = buckets.popMinimum();
        removed[v] = 1;
        order.push_back(v);
        for (int w : g.neighbors(v))
            if (!removed[w]) buckets.decrement(w);
    }
    return order;
}

std::vector<std::vector<int>> MaximalCliqueEnumerator::enumerate(const NeighborGraph& g) {
    const std::vector<int> order = degeneracyOrder(g);
    std::vector<int> position(g.size(), -1);
    for (std::size_t i = 0; i < order.size(); ++i) position[order[i]] = static_cast<int>(i);
    std::vector<std::vector<int>> cliques;
    for (int v : order) {
        SearchFrame root;
        root.clique = {v};
        for (int w : g.neighbors(v)) (position[w] > position[v] ? root.candidates : root.excluded).push_back(w);
        std::vector<SearchFrame> stack;
        stack.push_back(std::move(root));
        while (!stack.empty()) {
            SearchFrame frame = std::move(stack.back());
            stack.pop_back();
            if (frame.candidates.empty() && frame.excluded.empty()) {
                if (frame.clique.size() >= 2) {
                    std::sort(frame.clique.begin(), frame.clique.end());
                    cliques.push_back(std::move(frame.clique));
                }
                continue;
            }
            // pivot: the vertex of P u X with the most neighbours in P
            int pivot = -1;
            std::size_t best = 0;
            for (const auto* set : {&frame.candidates, &frame.excluded})
                for (int u : *set) {
                    const std::size_t c = intersect(g.neighbors(u), frame.candidates).size();
                    if (pivot < 0 || c > best) {
                        best = c;
                        pivot = u;
                    }
                }
            std::vector<int> branch;
            std::set_difference(frame.candidates.begin(), frame.candidates.end(), g.neighbors(pivot).begin(),
                                g.neighbors(pivot).end(), std::back_inserter(branch));
            std::vector<int> candidates = frame.candidates, excluded = frame.excluded;
            for (int u : branch) {
                SearchFrame child;
                child.clique = frame.clique;
                child.clique.push_back(u);
                child.candidates = intersect(candidates, g.neighbors(u));
                child.excluded = intersect(excluded, g.neighbors(u));
                stack.push_back(std::move(child));
                candidates.erase(std::lower_bound(candidates.begin(), candidates.end(), u));
                excluded.insert(std::lower_bound(excluded.begin(), excluded.end(), u), u);
            }
        }
    }
    std::sort(cliques.begin(), cliques.end());
    return cliques;
}

CliqueTable::CliqueTable(const SpatialDataset& data, const std::vector<std::vector<int>>& cliques)
    : rows_(static_cast<int>(cliques.size())), features_(data.featureCount()), instances_(data.size()),
      members_(static_cast<std::size_t>(cliques.size()) * data.featureCount(), kNone), rowsWithFeature_(data.featureCount()) {
    for (int c = 0; c < rows_; ++c)
        for (int o : cliques[c]) members_[static_cast<std::size_t>(c) * features_ + data.instance(o).feature] = o;
    for (int c = 0; c < rows_; ++c)
        for (int f = 0; f < features_; ++f)
            if (has(c, f)) rowsWithFeature_[f].push_back(c);
}

CliqueTable CliqueTable::build(const SpatialDataset& data, double distance) {
    const NeighborGraph graph(data, distance);
    return CliqueTable(data, MaximalCliqueEnumerator::enumerate(graph));
}

std::vector<int> CliqueTable::participatingInstances(const std::vector<int>& pattern) const {
    std::vector<char> mark(instances_, 0);
    for (int c = 0; c < rows_; ++c) {
        bool all = true;
        for (int f : pattern)
            if (!has(c, f)) {
                all = false;
                break;
            }
        if (!all) continue;
        for (int f : pattern) mark[at(c, f)] = 1;
    }
    std::vector<int> out;
    for (int i = 0; i < instances_; ++i)
        if (mark[i]) out.push_back(i);
    return out;
}

}  // namespace colocation
