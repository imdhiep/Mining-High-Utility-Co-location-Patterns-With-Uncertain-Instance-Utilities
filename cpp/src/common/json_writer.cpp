#include "common/json_writer.hpp"

#include <cmath>
#include <iomanip>
#include <limits>

namespace colocation {

namespace {

void writeNumber(std::ostringstream& out, double v) {
    if (std::isfinite(v)) out << v;
    else out << "null";
}

}  // namespace

JsonLineWriter::JsonLineWriter() { out_ << std::setprecision(std::numeric_limits<double>::max_digits10) << "{"; }

std::string JsonLineWriter::escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

void JsonLineWriter::key(const std::string& k) {
    out_ << (first_ ? "" : ", ") << "\"" << escape(k) << "\": ";
    first_ = false;
}

JsonLineWriter& JsonLineWriter::field(const std::string& k, const std::string& value) {
    key(k);
    out_ << "\"" << escape(value) << "\"";
    return *this;
}

JsonLineWriter& JsonLineWriter::field(const std::string& k, double value) {
    key(k);
    writeNumber(out_, value);
    return *this;
}

JsonLineWriter& JsonLineWriter::field(const std::string& k, long long value) {
    key(k);
    out_ << value;
    return *this;
}

JsonLineWriter& JsonLineWriter::field(const std::string& k, bool value) {
    key(k);
    out_ << (value ? "true" : "false");
    return *this;
}

JsonLineWriter& JsonLineWriter::field(const std::string& k, const std::vector<std::string>& values) {
    key(k);
    out_ << "[";
    for (std::size_t i = 0; i < values.size(); ++i) out_ << (i ? ", " : "") << "\"" << escape(values[i]) << "\"";
    out_ << "]";
    return *this;
}

JsonLineWriter& JsonLineWriter::field(const std::string& k, const std::vector<double>& values) {
    key(k);
    out_ << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        out_ << (i ? ", " : "");
        writeNumber(out_, values[i]);
    }
    out_ << "]";
    return *this;
}

JsonLineWriter& JsonLineWriter::field(const std::string& k, const std::vector<JsonLineWriter>& objects) {
    key(k);
    out_ << "[";
    for (std::size_t i = 0; i < objects.size(); ++i) out_ << (i ? ", " : "") << objects[i].str();
    out_ << "]";
    return *this;
}

std::string JsonLineWriter::str() const { return out_.str() + "}"; }

}  // namespace colocation
