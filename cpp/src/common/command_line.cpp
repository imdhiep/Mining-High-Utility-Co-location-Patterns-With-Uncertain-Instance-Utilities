#include "common/command_line.hpp"

#include <stdexcept>

namespace colocation {

CommandLine::CommandLine(int argc, char** argv, const std::set<std::string>& flags) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0) {
            positional_.push_back(arg);
            continue;
        }
        const std::string name = arg.substr(2);
        if (flags.count(name)) {
            options_[name] = "1";
        } else {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            options_[name] = argv[++i];
        }
    }
}

const std::string& CommandLine::positional(std::size_t i) const {
    if (i >= positional_.size()) throw std::invalid_argument("missing positional argument " + std::to_string(i + 1));
    return positional_[i];
}

double CommandLine::positionalDouble(std::size_t i) const { return std::stod(positional(i)); }
int CommandLine::positionalInt(std::size_t i) const { return std::stoi(positional(i)); }

std::string CommandLine::option(const std::string& name, const std::string& fallback) const {
    auto it = options_.find(name);
    return it == options_.end() ? fallback : it->second;
}

double CommandLine::option(const std::string& name, double fallback) const {
    auto it = options_.find(name);
    return it == options_.end() ? fallback : std::stod(it->second);
}

int CommandLine::option(const std::string& name, int fallback) const {
    auto it = options_.find(name);
    return it == options_.end() ? fallback : std::stoi(it->second);
}

unsigned long long CommandLine::optionUnsigned(const std::string& name, unsigned long long fallback) const {
    auto it = options_.find(name);
    return it == options_.end() ? fallback : std::stoull(it->second);
}

}  // namespace colocation
