// Command-line arguments: positional values followed by "--name value" options and "--flag" switches.
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

namespace colocation {

class CommandLine {
public:
    // flags: option names that take no value (e.g. "scores" for --scores)
    CommandLine(int argc, char** argv, const std::set<std::string>& flags = {});

    std::size_t positionalCount() const { return positional_.size(); }
    const std::string& positional(std::size_t i) const;
    double positionalDouble(std::size_t i) const;
    int positionalInt(std::size_t i) const;

    bool has(const std::string& name) const { return options_.count(name) > 0; }
    std::string option(const std::string& name, const std::string& fallback) const;
    double option(const std::string& name, double fallback) const;
    int option(const std::string& name, int fallback) const;
    unsigned long long optionUnsigned(const std::string& name, unsigned long long fallback) const;

private:
    std::vector<std::string> positional_;
    std::map<std::string, std::string> options_;
};

}  // namespace colocation
