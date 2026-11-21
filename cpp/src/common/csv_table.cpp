#include "common/csv_table.hpp"

#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace colocation {

std::vector<std::string> CsvTable::splitLine(const std::string& line, char delimiter) {
    std::vector<std::string> fields;
    std::string field;
    std::stringstream stream(line);
    while (std::getline(stream, field, delimiter)) fields.push_back(field);
    if (!line.empty() && line.back() == delimiter) fields.emplace_back();
    return fields;
}

CsvTable CsvTable::read(const std::string& path, char delimiter) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    CsvTable table;
    std::string line;
    bool headerRead = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (!headerRead) {
            table.header_ = splitLine(line, delimiter);
            headerRead = true;
        } else {
            table.rows_.push_back(splitLine(line, delimiter));
        }
    }
    return table;
}

int CsvTable::column(const std::string& name) const {
    for (std::size_t i = 0; i < header_.size(); ++i)
        if (header_[i] == name) return static_cast<int>(i);
    throw std::runtime_error("missing column " + name);
}

bool CsvTable::hasColumn(const std::string& name) const {
    for (const auto& h : header_)
        if (h == name) return true;
    return false;
}

CsvWriter::CsvWriter(const std::string& path, const std::vector<std::string>& header, char delimiter)
    : out_(path), delimiter_(delimiter) {
    if (!out_) throw std::runtime_error("cannot write " + path);
    out_ << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto& h : header) add(h);
    endRow();
}

void CsvWriter::separator() {
    if (rowStarted_) out_ << delimiter_;
    rowStarted_ = true;
}

CsvWriter& CsvWriter::add(const std::string& value) {
    separator();
    out_ << value;
    return *this;
}

CsvWriter& CsvWriter::add(double value) {
    separator();
    out_ << value;
    return *this;
}

CsvWriter& CsvWriter::add(long long value) {
    separator();
    out_ << value;
    return *this;
}

void CsvWriter::endRow() {
    out_ << '\n';
    rowStarted_ = false;
}

}  // namespace colocation
