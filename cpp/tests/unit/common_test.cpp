#include <cstdio>
#include <fstream>
#include <random>

#include "common/command_line.hpp"
#include "common/csv_table.hpp"
#include "common/json_line_parser.hpp"
#include "common/json_writer.hpp"
#include "common/utility_posterior.hpp"
#include "test_framework.hpp"

using namespace colocation;

namespace {

std::string tempPath(const std::string& name) { return "/tmp/colocation_unit_" + name; }

}  // namespace

TEST(csv_split_keeps_trailing_empty_field) {
    const auto f = CsvTable::splitLine("a,b,", ',');
    CHECK(f.size() == 3);
    CHECK(f[2].empty());
    CHECK(CsvTable::splitLine("x\ty", '\t').size() == 2);
}

TEST(csv_read_strips_carriage_returns_and_blank_lines) {
    const std::string path = tempPath("t.csv");
    {
        std::ofstream o(path);
        o << "Feature,LocX\r\n\r\nA,1.5\r\nB,2\r\n";
    }
    const CsvTable t = CsvTable::read(path);
    CHECK(t.size() == 2);
    CHECK(t.column("LocX") == 1);
    CHECK(t.rows()[1][1] == "2");
    CHECK(t.hasColumn("Feature"));
    CHECK_THROWS(t.column("Missing"));
    std::remove(path.c_str());
}

TEST(csv_writer_round_trips_doubles) {
    const std::string path = tempPath("w.csv");
    const double v = 0.1 + 0.2;
    {
        CsvWriter w(path, {"x", "name"});
        w.add(v).add("p");
        w.endRow();
    }
    const CsvTable t = CsvTable::read(path);
    CHECK(std::stod(t.rows()[0][0]) == v);
    CHECK(t.rows()[0][1] == "p");
    std::remove(path.c_str());
}

TEST(json_parser_decodes_top_level_values) {
    const std::string line = R"({"city": "São \"P\"", "n": -1.5e2, "ok": true, "none": null, "nested": {"a": 1, "b": [1, 2]}, "list": [3]})";
    JsonObject o = JsonLineParser(line).parseObject();
    CHECK(o["city"].isString());
    CHECK(o["city"].text == "S\xc3\xa3o \"P\"");
    CHECK(o["n"].isNumber());
    CHECK(o["n"].number == -150.0);
    CHECK(o["ok"].kind == JsonValue::Kind::Bool && o["ok"].boolean);
    CHECK(o["none"].kind == JsonValue::Kind::Null);
    CHECK(o["nested"].kind == JsonValue::Kind::Raw);
    const std::string raw = o["nested"].text;
    JsonObject inner = JsonLineParser(raw).parseObject();
    CHECK(inner["a"].number == 1.0);
    CHECK(o["list"].kind == JsonValue::Kind::Raw);
}

TEST(json_parser_rejects_malformed_lines) {
    const std::string bad = R"({"a": 1 "b": 2})";
    CHECK_THROWS(JsonLineParser(bad).parseObject());
}

TEST(json_writer_writes_one_line) {
    JsonLineWriter w;
    w.field("s", "a\"b").field("n", 3).field("x", 0.5).field("b", false).field("l", std::vector<std::string>{"P", "Q"});
    CHECK(w.str() == R"({"s": "a\"b", "n": 3, "x": 0.5, "b": false, "l": ["P", "Q"]})");
}

TEST(command_line_positionals_options_and_flags) {
    const char* args[] = {"prog", "data.csv", "0.5", "--bound", "mean", "--scores", "7"};
    CommandLine cl(7, const_cast<char**>(args), {"scores"});
    CHECK(cl.positionalCount() == 3);
    CHECK(cl.positional(0) == "data.csv");
    CHECK(cl.positionalDouble(1) == 0.5);
    CHECK(cl.positionalInt(2) == 7);
    CHECK(cl.option("bound", std::string("two")) == "mean");
    CHECK(cl.has("scores"));
    CHECK(cl.option("missing", 4) == 4);
    CHECK_THROWS(cl.positional(3));
    const char* bad[] = {"prog", "--bound"};
    CHECK_THROWS(CommandLine(2, const_cast<char**>(bad)));
}

TEST(posterior_file_round_trip) {
    std::mt19937_64 rng(3);
    std::normal_distribution<double> n01;
    UtilityPosterior p;
    for (int i = 0; i < 7; ++i) {
        p.mean.push_back(n01(rng));
        p.variance.push_back(std::fabs(n01(rng)));
    }
    p.bootstrap = BootstrapMatrix(4, 7);
    for (double& v : p.bootstrap.values()) v = n01(rng);
    p.normalizer = 12.25;
    const std::string path = tempPath("p.bin");
    UtilityPosteriorFile::write(path, p);
    const UtilityPosterior q = UtilityPosteriorFile::read(path);
    CHECK(q.mean == p.mean);
    CHECK(q.variance == p.variance);
    CHECK(q.bootstrap.samples() == 4 && q.bootstrap.instances() == 7);
    CHECK(q.bootstrap.values() == p.bootstrap.values());
    CHECK(q.normalizer == p.normalizer);
    UtilityPosterior empty;
    empty.mean = {1.0};
    empty.variance = {0.5};
    UtilityPosteriorFile::write(path, empty);
    CHECK(UtilityPosteriorFile::read(path).bootstrap.empty());
    std::remove(path.c_str());
}
