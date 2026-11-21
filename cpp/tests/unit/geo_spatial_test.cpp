#include <random>

#include "geo/utm_projection.hpp"
#include "spatial/clique_table.hpp"
#include "spatial/neighbor_graph.hpp"
#include "test_data.hpp"
#include "test_framework.hpp"

using namespace colocation;

TEST(utm_matches_pyproj_reference_points) {
    // pyproj 3.8 (EPSG:4326 -> EPSG:326ZZ), computed once and pasted here
    struct Point { int zone; double lon, lat, x, y; };
    const Point points[] = {
        {18, -74.006, 40.7128, 583959.372324085, 4507350.998243321},
        {18, -75.0, 0.0001, 500000.0, 11.053004611126683},
        {18, -77.9, 44.5, 269454.6731095505, 4931500.388984377},
        {54, 139.6917, 35.6895, 381622.23003942776, 3950298.9078813544},
        {11, -115.1398, 36.1699, 667304.0470087623, 4004396.314793694},
        {17, -79.3832, 43.6532, 630378.9952610959, 4834625.700882426},
        {12, -112.074, 33.4484, 400179.64778516354, 3701514.105575749},
        {54, 141.0, 30.0, 500000.00000000244, 3318785.3525812062},
    };
    for (const auto& p : points) {
        const PlanarPoint q = UtmProjection(p.zone).project(p.lon, p.lat);
        CHECK_NEAR(q.x, p.x, 1e-6);
        CHECK_NEAR(q.y, p.y, 1e-6);
    }
}

TEST(utm_zone_of_longitude) {
    CHECK(UtmProjection::zoneOfLongitude(-74.0) == 18);
    CHECK(UtmProjection::zoneOfLongitude(139.7) == 54);
    CHECK(UtmProjection::zoneOfLongitude(-180.0) == 1);
    CHECK(UtmProjection::zoneOfLongitude(-72.0) == 19);   // a zone border belongs to the zone on its east
}

TEST(neighbor_graph_equals_all_pairs) {
    std::mt19937_64 rng(1);
    for (int it = 0; it < 200; ++it) {
        const auto data = unit::randomDataset(rng, 5 + it % 60, 2 + it % 6, 100.0);
        const double d = 5.0 + (it % 7) * 6.0;
        const NeighborGraph g(data, d);
        std::set<std::pair<int, int>> got;
        for (int i = 0; i < g.size(); ++i)
            for (int j : g.neighbors(i))
                if (i < j) got.insert({i, j});
        CHECK(got == unit::bruteNeighbors(data, d));
        for (int i = 0; i < g.size(); ++i) CHECK(std::is_sorted(g.neighbors(i).begin(), g.neighbors(i).end()));
    }
}

TEST(neighbor_graph_keeps_exact_ties_and_has_no_same_feature_edges) {
    std::vector<SpatialInstance> inst = {{0, 0, 0}, {1, 3, 4}, {0, 6, 8}, {1, 0, 0.5}};
    const SpatialDataset data(inst, {"A", "B"});
    const NeighborGraph g(data, 5.0);
    CHECK(g.neighbors(0) == (std::vector<int>{1, 3}));   // distance exactly 5 is kept
    CHECK(g.neighbors(2) == (std::vector<int>{1}));
    CHECK(g.edgeCount() == 3);
    CHECK(NeighborGraph(data, 0.1).edgeCount() == 0);
}

TEST(maximal_cliques_are_cliques_maximal_and_cover_every_clique) {
    std::mt19937_64 rng(2);
    for (int it = 0; it < 150; ++it) {
        const auto data = unit::randomDataset(rng, 5 + it % 45, 2 + it % 6, 60.0);
        const double d = 8.0 + (it % 5) * 5.0;
        const NeighborGraph g(data, d);
        const auto cliques = MaximalCliqueEnumerator::enumerate(g);
        auto adjacent = [&](int a, int b) { return std::binary_search(g.neighbors(a).begin(), g.neighbors(a).end(), b); };
        for (const auto& c : cliques) {
            CHECK(c.size() >= 2);
            for (std::size_t a = 0; a < c.size(); ++a)
                for (std::size_t b = a + 1; b < c.size(); ++b) CHECK(adjacent(c[a], c[b]));
            for (int v = 0; v < g.size(); ++v) {
                if (std::binary_search(c.begin(), c.end(), v)) continue;
                bool extends = true;
                for (int u : c) extends = extends && adjacent(u, v);
                CHECK(!extends);
            }
        }
        CHECK(std::is_sorted(cliques.begin(), cliques.end()));
        // every edge lies in some maximal clique
        for (int i = 0; i < g.size(); ++i)
            for (int j : g.neighbors(i)) {
                bool covered = false;
                for (const auto& c : cliques)
                    covered = covered || (std::binary_search(c.begin(), c.end(), i) && std::binary_search(c.begin(), c.end(), j));
                CHECK(covered);
            }
    }
}

TEST(participating_instances_equal_row_instances) {
    std::mt19937_64 rng(3);
    for (int it = 0; it < 150; ++it) {
        const auto data = unit::randomDataset(rng, 5 + it % 45, 2 + it % 6, 60.0);
        const double d = 8.0 + (it % 5) * 5.0;
        const CliqueTable table = CliqueTable::build(data, d);
        for (const auto& kv : unit::brutePatterns(data, d)) {
            const std::vector<int> got = table.participatingInstances(kv.first);
            CHECK(std::set<int>(got.begin(), got.end()) == kv.second);
        }
    }
}

TEST(clique_table_handles_isolated_and_single_feature_data) {
    std::vector<SpatialInstance> inst = {{0, 0, 0}, {0, 1, 0}, {0, 2, 0}};
    const SpatialDataset one(inst, {"A"});
    const CliqueTable t = CliqueTable::build(one, 5.0);
    CHECK(t.rows() == 0);
}
