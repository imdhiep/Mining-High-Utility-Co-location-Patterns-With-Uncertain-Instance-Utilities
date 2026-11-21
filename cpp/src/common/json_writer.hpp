// Builds one JSON object as a single line (the result line every program prints for the experiment scripts).
#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace colocation {

class JsonLineWriter {
public:
    JsonLineWriter();
    JsonLineWriter& field(const std::string& key, const std::string& value);
    JsonLineWriter& field(const std::string& key, const char* value) { return field(key, std::string(value)); }
    JsonLineWriter& field(const std::string& key, double value);
    JsonLineWriter& field(const std::string& key, long long value);
    JsonLineWriter& field(const std::string& key, int value) { return field(key, static_cast<long long>(value)); }
    JsonLineWriter& field(const std::string& key, std::size_t value) { return field(key, static_cast<long long>(value)); }
    JsonLineWriter& field(const std::string& key, bool value);
    JsonLineWriter& field(const std::string& key, const std::vector<std::string>& values);
    JsonLineWriter& field(const std::string& key, const std::vector<double>& values);
    // an array of objects built by other writers
    JsonLineWriter& field(const std::string& key, const std::vector<JsonLineWriter>& objects);
    std::string str() const;

    static std::string escape(const std::string& s);

private:
    std::ostringstream out_;
    bool first_ = true;
    void key(const std::string& k);
};

}  // namespace colocation
