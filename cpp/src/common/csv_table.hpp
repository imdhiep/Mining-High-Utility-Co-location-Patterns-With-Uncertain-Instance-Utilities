// Reading and writing plain delimited text tables (no quoting: every file of the project holds numbers, identifiers
// and category names without the delimiter).
#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace colocation {

class CsvTable {
public:
    // reads a file whose first non-empty line is the header; trailing carriage returns are removed
    static CsvTable read(const std::string& path, char delimiter = ',');
    // splits one line; a trailing delimiter yields a final empty field
    static std::vector<std::string> splitLine(const std::string& line, char delimiter);

    const std::vector<std::string>& header() const { return header_; }
    const std::vector<std::vector<std::string>>& rows() const { return rows_; }
    std::size_t size() const { return rows_.size(); }
    // index of a column, throws std::runtime_error when the column is missing
    int column(const std::string& name) const;
    bool hasColumn(const std::string& name) const;

private:
    std::vector<std::string> header_;
    std::vector<std::vector<std::string>> rows_;
};

// Writes a table row by row; values are written with full double precision.
class CsvWriter {
public:
    CsvWriter(const std::string& path, const std::vector<std::string>& header, char delimiter = ',');
    CsvWriter& add(const std::string& value);
    CsvWriter& add(double value);
    CsvWriter& add(long long value);
    CsvWriter& add(int value) { return add(static_cast<long long>(value)); }
    void endRow();

private:
    std::ofstream out_;
    char delimiter_;
    bool rowStarted_ = false;
    void separator();
};

}  // namespace colocation
