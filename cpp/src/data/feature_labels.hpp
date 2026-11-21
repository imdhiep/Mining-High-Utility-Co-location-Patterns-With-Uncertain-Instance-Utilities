// One-character feature labels of the instance files: every miner of the project (and the authors' CQ-HUCPM and
// PSO-MUCP code) reads the feature of an instance from the first character of the Feature column.
#pragma once

#include <stdexcept>
#include <string>

namespace colocation {

class FeatureLabels {
public:
    // labels in the order they are given to features sorted by name
    static constexpr const char* kAlphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!#$%()*+-./:;<=>?@[]^_`{|}~";

    static int capacity() { return static_cast<int>(std::string(kAlphabet).size()); }
    static char label(int featureIndex) {
        if (featureIndex < 0 || featureIndex >= capacity()) throw std::out_of_range("too many features for one-character labels");
        return kAlphabet[featureIndex];
    }
};

}  // namespace colocation
