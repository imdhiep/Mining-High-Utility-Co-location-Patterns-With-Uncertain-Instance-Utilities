// Special functions not in the C++ standard library.
#pragma once

namespace colocation {

class SpecialFunctions {
public:
    static double digamma(double x);           // x > 0
    static double trigamma(double x);          // x > 0
    static double normalCdf(double z);
    static double normalPdf(double z);
};

}  // namespace colocation
